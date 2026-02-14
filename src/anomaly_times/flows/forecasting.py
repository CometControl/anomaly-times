from prefect import flow, task, get_run_logger
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
from ..core.reader import read_metric
from ..core.writer import write_metric

from ..core.reader import read_metric
from ..core.writer import write_metric
from ..models.utils import get_model_flow


from prefect_ray.task_runners import RayTaskRunner
import fsspec
from datetime import timezone




@flow(
    name="forecast_flow",
)
def forecast_flow(
    promql: str = "http_requests_total",
    lookback_minutes: int = 60,
    forecast_horizon_minutes: int = 60,
    tsdb_url: str = "http://victoria-metrics:8428",
    model_type: str = "lgbm",
    is_multivariate: bool = False,
    artifact_storage_path: Optional[str] = None, # e.g. s3://bucket/models/{promql_hash}
    fit_expiration_hours: int = 24
):
    """
    Orchestrates forecasting with stateful model management.
    """
    logger = get_run_logger()
    
    # Align to minute boundary for deterministic query steps
    end_time = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    start_time = end_time - timedelta(minutes=lookback_minutes)
    
    # 1. Fetch Context (Results in Panel Data: timestamp, unique_id, value)
    df = read_metric(
        query=promql,
        start_time=start_time, 
        end_time=end_time,
        url=tsdb_url
    )
    
    if df.empty:
        logger.warning("No data found for context. Skipping forecast.")
        return

    logger.info(f"Fetched {len(df)} rows for {df['unique_id'].nunique()} series.")

    logger.info(f"Fetched {len(df)} rows for {df['unique_id'].nunique()} series.")

    # 1. Resolve Model Flow via Auto-Discovery
    try:
        model_flow = get_model_flow(model_type)
    except ValueError as e:
        logger.error(str(e))
        return


    # 2. Dispatch to Sub-Flow
    logger.info(f"Dispatching to {model_type} sub-flow...")
    
    # We pass 'multivariate' to all. 
    # Sub-flows that don't need it will ignore it via **kwargs.
    # Sub-flows that need it (TimesNet) will use it.
    
    forecast_df = model_flow(
        context_df=df,
        horizon=forecast_horizon_minutes,
        confidence_level=0.9,
        storage_path=artifact_storage_path,
        fit_expiration_hours=fit_expiration_hours,
        multivariate=is_multivariate
    )

    if forecast_df.empty:
        logger.warning("Forecast returned empty dataframe.")
        return
    
    # 3. Write Forecast
    # We must write back with the original labels (present in unique_id)
    # The writer needs to handle 'unique_id' which is a JSON string of labels.
    
    # Forecast DF: [timestamp, unique_id, pred, lower, upper]
    
    # Write Pred
    write_metric(
        dataframe=forecast_df[['timestamp', 'unique_id', 'pred']], 
        metric_name="anomaly_pred", # Base name, labels come from unique_id
        url=f"{tsdb_url}/api/v1/import/csv"
    )
    # Write Bounds (Optional, maybe different metric names)
    write_metric(
        dataframe=forecast_df[['timestamp', 'unique_id', 'lower']], 
        metric_name="anomaly_lower",
        url=f"{tsdb_url}/api/v1/import/csv"
    )
    write_metric(
        dataframe=forecast_df[['timestamp', 'unique_id', 'upper']], 
        metric_name="anomaly_upper",
        url=f"{tsdb_url}/api/v1/import/csv"
    )
    
    logger.info("Forecast written successfully.")
