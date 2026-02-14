from datetime import datetime
import pandas as pd
from prometheus_api_client import PrometheusConnect, MetricRangeDataFrame
from prefect import task
from typing import Optional
import json

@task(name="read_prometheus_metric")
def read_metric(
    query: str,
    start_time: datetime,
    end_time: datetime,
    step: str = "1m",
    url: str = "http://localhost:8428",
    disable_ssl: bool = True
) -> pd.DataFrame:
    """
    Reads a metric from a Prometheus-compatible TSDB and returns a pandas DataFrame
    in 'Panel Data' format (Long format).
    
    Returns columns: [timestamp, unique_id, value] + other tags
    """
    prom = PrometheusConnect(url=url, disable_ssl=disable_ssl)
    
    metric_data = prom.get_metric_range_data(
        metric_name=query,
        start_time=start_time,
        end_time=end_time,
        chunk_size=None,
    )
    
    df = MetricRangeDataFrame(metric_data)
    
    if df.empty:
        return df

    # Standardize columns
    # MetricRangeDataFrame usually has: timestamp, value, + label columns
    # We want to create a 'unique_id' from the labels to distinguish series
    
    # 1. Reset index if timestamp is index
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        
    if 'timestamp' not in df.columns and 'ds' not in df.columns:
         # Try to find date col
         date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
         if date_cols:
             df = df.rename(columns={date_cols[0]: 'timestamp'})

    # 2. Generate unique_id
    # Filter out timestamp and value to get label cols
    label_cols = [c for c in df.columns if c not in ['timestamp', 'value', '__name__']]
    
    if not label_cols:
        df['unique_id'] = 'series_0'
    else:
        # Create a string representation of labels as unique_id
        # e.g. {instance="host1", job="node"}
        def make_id(row):
            labels = {k: row[k] for k in label_cols}
            return json.dumps(labels, sort_keys=True)
            
        df['unique_id'] = df.apply(make_id, axis=1)

    return df[['timestamp', 'unique_id', 'value']]
