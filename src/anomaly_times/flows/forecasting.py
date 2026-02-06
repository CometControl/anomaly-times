from prefect import flow, task, get_run_logger
from datetime import datetime, timedelta
import pandas as pd
from ..core.reader import read_metric
from ..core.writer import write_metric
from ..models.tsfm import ChronosModel
# from ..models.nixtla import NixtlaModel
from prefect_ray.task_runners import RayTaskRunner

import fsspec
from datetime import timezone

@task
def load_and_predict(
    context_df: pd.DataFrame,
    model_type: str = "arima",
    horizon: int = 60,
    confidence_level: float = 0.9,
    multivariate: bool = False,
    storage_path: str = None, # e.g. s3://bucket/model.pkl
    fit_expiration_hours: int = 24
) -> pd.DataFrame:
    """
    Task to load model and predict.
    Implements Check-Load-Or-Fit-Save logic.
    """
    logger = get_run_logger()
    logger.info(f"Predicting with {model_type} (Multivariate: {multivariate})")
    
    params = {
        "freq": "1min", 
        "multivariate": multivariate,
        "horizon": horizon 
    }

    # Model Class Selector
    if model_type == "chronos":
        from ..models.tsfm.chronos import ChronosModel
        ModelClass = ChronosModel
    elif model_type == "arima":
        from ..models.nixtla.arima import ArimaModel
        ModelClass = ArimaModel
    elif model_type == "timesnet":
        from ..models.nixtla.timesnet import TimesNetModel
        ModelClass = TimesNetModel
    elif model_type == "lgbm":
        from ..models.nixtla.lgbm import LgbmModel
        ModelClass = LgbmModel
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = None
    should_fit = True
    
    # 1. Check Storage
    if storage_path:
        try:
            fs, fs_path = fsspec.core.url_to_fs(storage_path)
            if fs.exists(fs_path):
                # Check Expiration
                info = fs.info(fs_path)
                mtime = info['mtime'] # usually timestamp or datetime
                # Ensure UTC
                if isinstance(mtime, float) or isinstance(mtime, int):
                    last_modified = datetime.fromtimestamp(mtime, tz=timezone.utc)
                else:
                    last_modified = mtime.replace(tzinfo=timezone.utc)
                
                age_hours = (datetime.now(timezone.utc) - last_modified).total_seconds() / 3600
                
                if age_hours < fit_expiration_hours:
                    logger.info(f"Found valid model at {storage_path} (Age: {age_hours:.1f}h). Loading...")
                    model = ModelClass.load(storage_path)
                    # For some models (Chronos), we still need context
                    if hasattr(model, 'context_df'): # If model stores context, update it?
                        # Actually 'fit' updates context. If loaded, we might need to supply context for modify predict?
                        # Standard interface: predict takes df. 
                        pass 
                    should_fit = False
                else:
                    logger.info(f"Model at {storage_path} expired (Age: {age_hours:.1f}h). Re-fitting...")
            else:
                logger.info(f"No existing model at {storage_path}. Fitting new...")
        except Exception as e:
            logger.warning(f"Error checking storage: {e}. Proceeding to fit.")
            
    # 2. Fit (if needed)
    if should_fit:
        logger.info("Fitting model...")
        model = ModelClass(params=params) # Create new
        model.fit(context_df)
        
        # Save if path provided
        if storage_path:
            logger.info(f"Saving model to {storage_path}...")
            try:
                model.save(storage_path)
            except Exception as e:
                logger.error(f"Failed to save model: {e}")

    # 3. Predict
    # Even if loaded, some models (Chronos) use the 'df' passed to predict as context
    forecast = model.predict(context_df, horizon=horizon, confidence_level=confidence_level)
    return forecast

@flow(
    name="forecast_flow",
    task_runner=RayTaskRunner()
)
def forecast_flow(
    promql: str = "http_requests_total",
    lookback_minutes: int = 60,
    forecast_horizon_minutes: int = 60,
    tsdb_url: str = "http://victoria-metrics:8428",
    model_type: str = "lgbm",
    is_multivariate: bool = False,
    artifact_storage_path: str = None, # e.g. s3://bucket/models/{promql_hash}
    fit_expiration_hours: int = 24
):
    """
    Orchestrates forecasting with stateful model management.
    """
    logger = get_run_logger()
    
    end_time = datetime.now()
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

    # 2. Predict
    forecast_df = load_and_predict(
        context_df=df,
        model_type=model_type,
        horizon=forecast_horizon_minutes,
        confidence_level=0.9,
        multivariate=is_multivariate,
        storage_path=artifact_storage_path,
        fit_expiration_hours=fit_expiration_hours
    )
    
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
