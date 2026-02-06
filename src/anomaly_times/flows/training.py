from prefect import flow, task, get_run_logger
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd
from ..core.reader import read_metric
# from ..models.nixtla import NixtlaModel
# from ..models.tsfm import ChronosModel

@flow(name="train_model_flow")
def train_model_flow(
    metric_name: str = "http_requests_total",
    model_type: str = "nixtla",
    hyperparameters: Optional[Dict[str, Any]] = None,
    tsdb_url: str = "http://victoria-metrics:8428",
    lookback_days: int = 30
):
    """
    Flow for training or fine-tuning models.
    
    This is a TEMPLATE flow. Users can trigger this via API with custom:
    - metric_name: Which series to train on.
    - model_type: 'nixtla', 'chronos', 'timesfm', etc.
    - hyperparameters: A dict of model-specific params (e.g., {'learning_rate': 0.01}).
    """
    logger = get_run_logger()
    hyperparameters = hyperparameters or {}
    logger.info(f"Starting training for {metric_name} using {model_type} with params: {hyperparameters}")
    
    # 1. Fetch History
    end_time = datetime.now()
    start_time = end_time - pd.Timedelta(days=lookback_days)
    
    df = read_metric(query=metric_name, start_time=start_time, end_time=end_time, url=tsdb_url)
    
    if df.empty:
        logger.warning("No data found for training.")
        return

    # 2. Train / Fine-tune Logic
    # if model_type == 'nixtla':
    #     model = NixtlaModel(params=hyperparameters)
    #     model.fit(df)
    #     # save model artifact...
    # elif model_type == 'chronos':
    #     # Fine-tune logic...
    #     pass
    
    logger.info("Training completed (Simulated).")
