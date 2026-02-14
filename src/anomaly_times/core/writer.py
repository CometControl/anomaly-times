import pandas as pd
import requests
import io
import json
from prefect import task
from typing import Optional

@task(name="write_prometheus_metric")
def write_metric(
    dataframe: pd.DataFrame,
    metric_name: str,
    url: str = "http://localhost:8428/api/v1/import/csv",
    extra_labels: Optional[dict] = None
) -> None:
    """
    Writes a pandas DataFrame to a Prometheus-compatible TSDB (VictoriaMetrics) via CSV import.
    Handles 'unique_id' column by parsing it as JSON labels if present.
    """
    if dataframe.empty:
        print("Empty dataframe, skipping write.")
        return

    df = dataframe.copy()
    
    # ensure timestamp
    if isinstance(df.index, pd.DatetimeIndex):
        df['timestamp'] = df.index.astype('int64') // 10**6 
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**6
    else:
        raise ValueError("DataFrame must have a datetime index or 'timestamp' column")

    # Determine value column
    value_col = None
    if 'y' in df.columns: value_col = 'y'
    elif 'value' in df.columns: value_col = 'value'
    elif 'pred' in df.columns: value_col = 'pred' # forecasting result
    elif 'anomaly_score' in df.columns: value_col = 'anomaly_score'
    else:
        # fallback
        for col in df.columns:
            if col not in ['timestamp', 'unique_id']:
                value_col = col
                break
    
    if not value_col:
        # If still null, maybe it is 'value' but checking failed?
        if 'value' in df.columns: value_col = 'value'
        else: raise ValueError("Could not determine value column")

    # Prepare export DataFrame
    export_df = pd.DataFrame()
    export_df['timestamp'] = df['timestamp']
    export_df['value'] = df[value_col]
    
    # Handle Labels from 'unique_id'
    if 'unique_id' in df.columns:
        # unique_id is expected to be a JSON string of labels
        # e.g. {"host": "h1", "job": "j1"}
        # We need to expand this into columns
        
        # Helper to safely parse
        def parse_labels(uid):
            try:
                return json.loads(uid)
            except:
                return {"series_id": str(uid)}
                
        # Apply and expand
        # This might be slow for huge dfs, but fine for batch forecasts
        label_dicts = df['unique_id'].apply(parse_labels).tolist()
        labels_df = pd.DataFrame(label_dicts)
        
        # Concatenate labels to export_df
        # Reset index to match
        export_df = pd.concat([export_df.reset_index(drop=True), labels_df.reset_index(drop=True)], axis=1)

    # Handle explicit Extra Labels
    if extra_labels:
        for k, v in extra_labels.items():
            export_df[k] = v
            
    # Add metric name
    export_df['__name__'] = metric_name

    # Convert to CSV buffer
    csv_buffer = io.StringIO()
    export_df.to_csv(csv_buffer, index=False, header=False)
    
    # Construct Format String for VM
    # format=1:time:unix_ms,2:metric:value,3:label:labelname...
    
    # Columns: timestamp(1), value(2), ...labels...
    format_parts = ["1:time:unix_ms", "2:metric:value"]
    col_idx = 3
    for col in export_df.columns:
        if col not in ['timestamp', 'value']:
            format_parts.append(f"{col_idx}:label:{col}")
            col_idx += 1
            
    format_str = ",".join(format_parts)
    params = {'format': format_str}
    
    try:
        response = requests.post(url, data=csv_buffer.getvalue(), params=params)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to write data: {e}")
        # raise e # Don't crash flow on write fail?
