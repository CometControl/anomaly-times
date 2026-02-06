import pandas as pd
import numpy as np
from prefect import task

@task(name="calculate_anomaly_score")
def calculate_anomaly_score(
    realtime_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    timestamp_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    Calculates anomaly score primarily based on deviation from forecast confidence bounds.
    VMAnomaly Style: Score between 0.0 and 1.0 is "normal", > 1.0 is "anomalous".
    
    Formula idea:
      If inside bounds (lower <= actual <= upper):
         score = |actual - pred| / (0.5 * (upper - lower))  --> simplistic mapping to [0, 1]
         We want 0 at 'pred', and 1 at 'bounds'.
      If outside bounds:
         score = 1.0 + distance_from_nearest_bound / (0.5 * width)
         
    Args:
        realtime_df: DataFrame with 'value' and 'timestamp'.
        forecast_df: DataFrame with 'pred', 'lower', 'upper', and 'timestamp'.
        
    Returns:
        pd.DataFrame: DataFrame with 'anomaly_score' and 'timestamp'.
    """
    if realtime_df.empty or forecast_df.empty:
        return pd.DataFrame()

    # Align DataFrames on timestamp
    # Ensure indexes are timestamps
    realtime_df = realtime_df.set_index(timestamp_col) if timestamp_col in realtime_df.columns else realtime_df
    forecast_df = forecast_df.set_index(timestamp_col) if timestamp_col in forecast_df.columns else forecast_df
    
    # Inner join to score only overlapping timestamps
    merged = realtime_df[['value']].join(forecast_df[['pred', 'lower', 'upper']], how='inner')
    
    if merged.empty:
        return pd.DataFrame()

    def score_row(row):
        actual = row['value']
        pred = row['pred']
        lower = row['lower']
        upper = row['upper']
        
        # Avoid division by zero
        width = upper - lower
        if width <= 1e-9:
             # Fallback if no interval width: simplistic deviation
             return abs(actual - pred)

        half_width = width / 2.0
        
        # If actual is exactly predicted, score 0
        deviation = abs(actual - pred)
        
        # If deviation == half_width (i.e. at bound), score should be 1.0
        score = deviation / half_width
        
        return score

    merged['anomaly_score'] = merged.apply(score_row, axis=1)
    
    # Reset index to return timestamp column
    result = merged[['anomaly_score']].reset_index()
    return result
