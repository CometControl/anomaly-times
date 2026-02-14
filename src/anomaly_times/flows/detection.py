from prefect import flow, get_run_logger
from datetime import datetime, timedelta, timezone
from typing import Optional
import pandas as pd
from ..core.reader import read_metric
from ..core.writer import write_metric
from ..core.anomaly import calculate_anomaly_score

@flow(name="detect_anomalies_flow")
def detect_anomalies_flow(
    metric_name: str = "http_requests_total", # Keeping param name for compatibility, but acts as promql_query
    promql: Optional[str] = None, # Explicit alias
    tsdb_url: str = "http://victoria-metrics:8428"
):
    """
    Runs recently to detect anomalies.
    Supports 'promql' input.
    Matches Realtime vs Forecast using 'unique_id' (labels).
    """
    logger = get_run_logger()
    query = promql or metric_name
    
    # Align to minute boundary for deterministic query steps
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    lookback = timedelta(minutes=5)
    start = now - lookback
    
    # 1. Read Realtime
    # Returns: [timestamp, unique_id, value]
    real_df = read_metric(query=query, start_time=start, end_time=now, url=tsdb_url)
    
    # 2. Read Forecast (pred, lower, upper)
    # The FORECAST was written with metric name "anomaly_pred" (and lower/upper) 
    # BUT applied to the *same labels* as the original series.
    # So we need to query 'anomaly_pred{original_labels...}'
    # Or simpler: Query 'anomaly_pred' and filter by unique_id logic? 
    # NO, we can't query all anomaly_preds. 
    # We ideally need to query 'anomaly_pred' using same label selectors as 'query'.
    # This is tricky if 'query' is complex (e.g. rate(x[5m])).
    # Assumption: The user provides a basic selector in 'query' like 'http_req{app="foo"}'
    # We can perform string manipulation or just query *all* anomaly_pred and merge?
    # Better: Query `anomaly_pred` filtering by the *exact sets of labels* we care about?
    # For MVP: Query 'anomaly_pred' and we join on unique_id. 
    # Note: unique_id generation needs to be deterministic.
    
    # To limit scope, we can try to inject the selectors from `query` into `anomaly_pred`.
    # E.g. query='http_total{job="a"}' -> forecast_query='anomaly_pred{job="a"}'
    # This requires 'promql_parser'. 
    
    # ALTERNATIVE: Just read ALL `anomaly_pred` and join. If usage is huge, this is bad.
    # But for "Per Deployment", the Deployment has a specific query.
    # If we use "Simple Selector" assumption:
    # We assume 'query' has selectors that we can apply to 'anomaly_pred'.
    
    # Simplest MVP workaround: Data join.
    # Read anomaly_pred, anomaly_lower, anomaly_upper.
    # Note: This reads GLOBAL forecasts. In production, need label filtering.
    
    pred_df = read_metric(query="anomaly_pred", start_time=start, end_time=now, url=tsdb_url)
    lower_df = read_metric(query="anomaly_lower", start_time=start, end_time=now, url=tsdb_url)
    upper_df = read_metric(query="anomaly_upper", start_time=start, end_time=now, url=tsdb_url)
    
    if real_df.empty or pred_df.empty:
        logger.info("Insufficient data.")
        return

    # Normalize Columns
    pred_df = pred_df.rename(columns={'value': 'pred'})
    lower_df = lower_df.rename(columns={'value': 'lower'})
    upper_df = upper_df.rename(columns={'value': 'upper'})
    
    # Merge Forecast Parts
    # Join on [timestamp, unique_id]
    forecast_full = pred_df.merge(lower_df, on=['timestamp', 'unique_id'], how='outer')\
                           .merge(upper_df, on=['timestamp', 'unique_id'], how='outer')
    
    # 3. Join Realtime with Forecast
    # Join on [timestamp, unique_id]
    merged = real_df.merge(forecast_full, on=['timestamp', 'unique_id'], how='inner')
    
    if merged.empty:
        logger.warning("No overlapping timestamps/IDs between Realtime and Forecast.")
        return

    # 4. Calculate Score
    # We iterate by unique_id to calculate scores per series? 
    # Or just vectorize if calculate_anomaly_score supports it.
    
    # Assuming calculate_anomaly_score is vectorized or we wrap:
    # Actually calculate_anomaly_score in core/anomaly.py expects 'value', 'pred', 'lower', 'upper' cols.
    # It should work on the full merged DF.
    
    score_df = calculate_anomaly_score(
        realtime_df=merged, # containing 'value'
        forecast_df=merged  # containing 'pred', 'lower', 'upper'
    )
    
    if score_df.empty:
        return
        
    # score_df has 'anomaly_score' and valid index/timestamp?
    # calculate_anomaly_score returns DF with 'anomaly_score' column and index.
    # We need to preserve 'unique_id'.
    
    # Quick fix: calculate_anomaly_score implementation check needed?
    # Let's assume it returns a Series or DF aligned with input.
    # If it returns new DF, we ensure unique_id is preserved.
    # Update: calculate_anomaly_score implementation in 'core/anomaly.py' uses simple vector ops.
    # So we can just add the column to 'merged'.
    
    merged['anomaly_score'] = calculate_anomaly_score(merged, merged)['anomaly_score']

    # 5. Write Score
    # We write 'anomaly_score' metric, preserving unique_id labels from 'merged'
    write_metric(
        dataframe=merged[['timestamp', 'unique_id', 'anomaly_score']],
        metric_name="anomaly_score",
        url=f"{tsdb_url}/api/v1/import/csv"
    )
    
    logger.info(f"Anomaly scores written for {len(merged)} points.")
