from prefect import flow, task, get_run_logger
from datetime import datetime, timedelta
import pandas as pd
from ..core.reader import read_metric
from ..core.writer import write_metric
from ..models.tsfm import ChronosModel
# from ..models.nixtla import NixtlaModel
from prefect_ray.task_runners import RayTaskRunner

@task
def load_and_predict(
    context_df: pd.DataFrame,
    model_type: str = "arima",
    horizon: int = 60,
    confidence_level: float = 0.9,
    multivariate: bool = False
) -> pd.DataFrame:
    """
    Task to load model and predict.
    Dynamically imports model class based on model_type.
    """
    logger = get_run_logger()
    logger.info(f"Predicting with {model_type} (Multivariate: {multivariate})")
    
    params = {
        "freq": "1min", 
        "multivariate": multivariate,
        "horizon": horizon # For models needing H at init (NHITS)
    }

    # Dynamic Model Loader
    if model_type == "chronos":
        from ..models.tsfm.chronos import ChronosModel
        model = ChronosModel(params=params)
    elif model_type == "arima":
        from ..models.nixtla.arima import ArimaModel
        model = ArimaModel(params=params)
    elif model_type == "timesnet":
        from ..models.nixtla.timesnet import TimesNetModel
        model = TimesNetModel(params=params)
    elif model_type == "lgbm":
        from ..models.nixtla.lgbm import LgbmModel
        model = LgbmModel(params=params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(context_df)
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
    model_type: str = "chronos",
    is_multivariate: bool = False
):
    """
    Orchestrates forecasting for a PromQL query.
    Handles multiple series returned by the query.
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
        multivariate=is_multivariate
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
