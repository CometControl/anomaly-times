import fsspec
from datetime import datetime, timezone
from prefect import task, get_run_logger
import pandas as pd
from typing import Type
from .base import BaseModel
import importlib

def get_model_flow(model_type: str):
    """
    Dynamically discovers and imports the flow for the given model_type.
    Searches in known subpackages ['nixtla', 'tsfm'].
    Expects module named '{model_type}.py' and flow/function named '{model_type}_flow'.
    """
    search_paths = ['nixtla', 'tsfm']
    
    for subpkg in search_paths:
        try:
            # Attempt import: src.anomaly_times.models.{subpkg}.{model_type}
            module_path = f"..{subpkg}.{model_type}"
            # import_module needs absolute path or relative with package arg
            # We are in src.anomaly_times.models.utils, so package is src.anomaly_times.models
            module = importlib.import_module(module_path, package=__package__)
            
            # Look for flow function
            flow_name = f"{model_type}_flow"
            if hasattr(module, flow_name):
                return getattr(module, flow_name)
        except ImportError:
            continue
            
    raise ValueError(f"Model '{model_type}' not found in subpackages {search_paths}. Ensure '{model_type}.py' exists and contains '{model_type}_flow'.")


@task
def run_stateful_model(
    model_class: Type[BaseModel],
    context_df: pd.DataFrame,
    params: dict,
    horizon: int,
    confidence_level: float,
    storage_path: str = None,
    fit_expiration_hours: int = 24
) -> pd.DataFrame:
    """
    Executes the Check-Load-Or-Fit-Save logic for any BaseModel.
    """
    logger = get_run_logger()
    model = None
    should_fit = True
    
    # 1. Check Storage
    if storage_path:
        try:
            fs, fs_path = fsspec.core.url_to_fs(storage_path)
            if fs.exists(fs_path):
                # Check Expiration
                info = fs.info(fs_path)
                mtime = info.get('mtime', 0)
                
                # Ensure UTC
                if isinstance(mtime, (float, int)):
                    last_modified = datetime.fromtimestamp(mtime, tz=timezone.utc)
                else:
                    last_modified = mtime.replace(tzinfo=timezone.utc)
                
                age_hours = (datetime.now(timezone.utc) - last_modified).total_seconds() / 3600
                
                if age_hours < fit_expiration_hours:
                    logger.info(f"Found valid model at {storage_path} (Age: {age_hours:.1f}h). Loading...")
                    model = model_class.load(storage_path)
                    should_fit = False
                else:
                    logger.info(f"Model at {storage_path} expired (Age: {age_hours:.1f}h). Re-fitting...")
            else:
                logger.info(f"No existing model at {storage_path}. Fitting new...")
        except Exception as e:
            logger.warning(f"Error checking storage: {e}. Proceeding to fit.")
            
    # 2. Fit (if needed)
    if should_fit:
        logger.info(f"Fitting {model_class.__name__}...")
        model = model_class(params=params)
        model.fit(context_df)
        
        # Save if path provided
        if storage_path:
            logger.info(f"Saving model to {storage_path}...")
            try:
                model.save(storage_path)
            except Exception as e:
                logger.error(f"Failed to save model: {e}")

    # 3. Predict
    return model.predict(context_df, horizon=horizon, confidence_level=confidence_level)
