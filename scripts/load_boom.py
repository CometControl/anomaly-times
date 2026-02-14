import os
import time
import json
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from huggingface_hub import hf_hub_download
import pyarrow.ipc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from datetime import timezone

VM_IMPORT_URL = os.getenv("VM_IMPORT_URL", "http://localhost:8428/api/v1/import")

def load_boom_series(repo_id="Datadog/BOOM", series_id="ds-0-T"):
    """
    Downloads and loads a specific series from the BOOM dataset.
    """
    filename = f"{series_id}/data-00000-of-00001.arrow"
    logger.info(f"Downloading {filename} from {repo_id}...")
    try:
        local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        
        # Read arrow file
        with pyarrow.ipc.open_stream(local_path) as reader:
            df = reader.read_all().to_pandas()
            
        return df
    except Exception as e:
        logger.error(f"Failed to load series {series_id}: {e}")
        raise

def parse_frequency(freq_str):
    """
    Parses pandas frequency string to seconds.
    Rough approximation for standard frequencies.
    """
    if not isinstance(freq_str, str):
        return 60

    freq_str = freq_str.upper()
    if freq_str == 'T' or freq_str == 'MIN':
        return 60
    elif freq_str == 'H':
        return 3600
    elif freq_str == 'D':
        return 86400
    elif freq_str == 'S':
        return 1
    # Handle multiples like '5T'
    if freq_str.endswith('T') or freq_str.endswith('MIN'):
        return int(freq_str[:-1]) * 60 if len(freq_str) > 1 else 60
    if freq_str.endswith('S'):
        return int(freq_str[:-1]) if len(freq_str) > 1 else 1
        
    logger.warning(f"Unknown frequency {freq_str}, defaulting to 60s")
    return 60

def ingest_to_vm(df, series_id, metric_base_name="boom_metric"):
    """
    Ingests the dataframe content to VictoriaMetrics.
    Shifts timestamps so the last point is 'now'.
    Sends everything in one go (or larger batches) per series file.
    """
    all_vm_timestamps = []
    all_vm_values = []
    all_metric_meta = []

    for index, row in df.iterrows():
        start_time = pd.Timestamp(row['start'])
        freq_str = row['freq']
        values = row['target']
        item_id = row['item_id']
        
        step_seconds = parse_frequency(freq_str)
        step_seconds = float(step_seconds)
        
        if isinstance(values, list):
            values = np.array(values)
            
        logger.info(f"Values type: {type(values)}")
        if hasattr(values, "shape"):
            logger.info(f"Values shape: {values.shape}")
            
        # Handle multivariate
        is_multivariate = False
        if values.ndim > 1:
            is_multivariate = True
        elif values.ndim == 1 and len(values) > 0:
             # Check if elements are arrays/lists (ragged)
            first_elem = values[0]
            if isinstance(first_elem, (list, np.ndarray, pd.Series)):
                 is_multivariate = True

        if is_multivariate:
            num_variates = len(values)
            logger.info(f"Detected multivariate series with {num_variates} variates (shape {values.shape})")
            for i in range(num_variates):
                variate_values = values[i]
                ts, vals, meta = process_single_series(variate_values, start_time, step_seconds, freq_str, metric_base_name, series_id, item_id, variate_idx=i)
                if ts:
                    all_vm_timestamps.append(ts)
                    all_vm_values.append(vals)
                    all_metric_meta.append(meta)
        else:
            ts, vals, meta = process_single_series(values, start_time, step_seconds, freq_str, metric_base_name, series_id, item_id)
            if ts:
                all_vm_timestamps.append(ts)
                all_vm_values.append(vals)
                all_metric_meta.append(meta)
                
    # Send all collected series in one batch request
    if all_vm_timestamps:
        send_batch_to_vm(all_vm_timestamps, all_vm_values, all_metric_meta)

def process_single_series(values, start_time, step_seconds, freq_str, metric_base_name, series_id, item_id, variate_idx=None):
    num_points = len(values)
    
    # Calculate time range
    # Center around NOW
    # Center around NOW, but align to nearest minute for clean timestamps
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    duration = timedelta(seconds=step_seconds * num_points)
    shifted_start = now - (duration / 2)
    
    logger.info(f"Processing series {item_id}. Points: {num_points}. Freq: {freq_str} ({step_seconds}s). Start: {shifted_start}")
    
    current_ts = shifted_start.timestamp()
    
    vm_timestamps = []
    vm_values = []
    
    # Check for gaps/NaNs
    nan_count = 0
    
    for v in values:
        # Check if v is scalar
        if hasattr(v, "__len__"):
             try:
                v = v[0] 
             except IndexError:
                v = np.nan

        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            ts_ms = int(current_ts * 1000)
            vm_timestamps.append(ts_ms)
            vm_values.append(float(v))
        else:
            nan_count += 1
        
        current_ts += step_seconds
    
    if nan_count > 0:
        logger.warning(f"Series {item_id} has {nan_count} NaNs/missing values out of {num_points} points.")

    metric_name = f"{metric_base_name}_{series_id.replace('-', '_')}"
    metric_labels = {
        "__name__": metric_name,
        "item_id": str(item_id),
        "original_start": str(start_time),
        "source": "boom_dataset"
    }
    if variate_idx is not None:
        metric_labels["variate"] = str(variate_idx)
        
    return vm_timestamps, vm_values, metric_labels

def send_batch_to_vm(timestamps_list, values_list, metric_labels_list):
    # Construct newline delimited JSON
    payload_lines = []
    total_points = 0
    
    for i in range(len(timestamps_list)):
        item = {
            "metric": metric_labels_list[i],
            "values": values_list[i],
            "timestamps": timestamps_list[i]
        }
        payload_lines.append(json.dumps(item))
        total_points += len(values_list[i])
        
    payload_str = "\n".join(payload_lines)
    
    logger.info(f"Sending batch with {len(timestamps_list)} series to VM...")
    try:
        response = requests.post(VM_IMPORT_URL, data=payload_str)
        if response.status_code != 204:
             logger.error(f"Failed to import data batch: {response.text}")
        else:
            logger.info(f"Successfully imported batch of {len(timestamps_list)} series ({total_points} total points).")
    except Exception as e:
        logger.error(f"Error sending request to VM: {e}")

if __name__ == "__main__":
    series_ids = ["ds-0-T", "ds-1-T"]
    
    # Check if VM is up
    try:
        requests.get(VM_IMPORT_URL.replace("/api/v1/import", "/health"))
    except Exception:
        logger.warning(f"VictoriaMetrics might not be reachable at {VM_IMPORT_URL}. Make sure it's running.")

    for series_id in series_ids:
        try:
            df = load_boom_series(series_id=series_id)
            ingest_to_vm(df, series_id)
        except Exception as e:
            logger.error(f"Failed to process {series_id}: {e}")
