from ..base import BaseModel
import pandas as pd
import torch
from typing import Dict, Any, Optional
from chronos import ChronosPipeline

class ChronosModel(BaseModel):
    """
    Wrapper for Amazon Chronos.
    Supports Batch/Panel processing for multiple 'unique_id' series.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.model_name = params.get('model_name', "amazon/chronos-t5-tiny")
        self.device_map = params.get('device_map', "cpu") 
        
        self.pipeline = ChronosPipeline.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            torch_dtype=torch.bfloat16 if self.device_map == "cuda" else torch.float32,
        )
        
    def save(self, path: str) -> None:
        # Zero-shot model doesn't need saving fitted weights
        # But to satisfy interface for stateful flow, we create a marker
        pass

    @classmethod
    def load(cls, path: str) -> 'ChronosModel':
        # Just re-init, ignoring path content
        return cls()

    def fit(self, df: pd.DataFrame) -> None:
        self.context_df = df.copy()

    def predict(self, df: pd.DataFrame, horizon: int, confidence_level: Optional[float] = None) -> pd.DataFrame:
        context = df if not df.empty else getattr(self, 'context_df', pd.DataFrame())
        
        if context.empty:
            raise ValueError("No context data provided.")

        # Group by unique_id to form batch
        unique_ids = context['unique_id'].unique()
        batch_context = []
        
        for uid in unique_ids:
            uid_df = context[context['unique_id'] == uid].sort_values('timestamp')
            # Extract values as tensor
            val_col = 'value' if 'value' in uid_df.columns else uid_df.columns[2]
            batch_context.append(torch.tensor(uid_df[val_col].values))

        # Predict Batch
        num_samples = self.params.get('num_samples', 20)
        forecast = self.pipeline.predict(
            batch_context,
            prediction_length=horizon,
            num_samples=num_samples,
        )
        # forecast shape: [num_series, num_samples, horizon]
        
        results = []
        
        # Process each series result
        for idx, uid in enumerate(unique_ids):
            forecast_torch = forecast[idx] 
            
            median_pred = torch.quantile(forecast_torch, 0.5, dim=0).numpy()
            lower_bound = median_pred
            upper_bound = median_pred
            
            if confidence_level:
                alpha = 1.0 - confidence_level
                q_low = alpha / 2.0
                q_high = 1.0 - q_low
                lower_bound = torch.quantile(forecast_torch, q_low, dim=0).numpy()
                upper_bound = torch.quantile(forecast_torch, q_high, dim=0).numpy()

            # Time index logic (per series)
            uid_df = context[context['unique_id'] == uid].sort_values('timestamp')
            last_ts = pd.to_datetime(uid_df['timestamp'].iloc[-1])
            freq = self.params.get('freq', '1min')
            future_dates = pd.date_range(start=last_ts, periods=horizon+1, freq=freq)[1:]
            
            series_res = pd.DataFrame({
                'timestamp': future_dates,
                'unique_id': uid,
                'pred': median_pred,
                'lower': lower_bound,
                'upper': upper_bound
            })
            results.append(series_res)
            
        return pd.concat(results)
