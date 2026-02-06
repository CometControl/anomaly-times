from ..base import BaseModel
import pandas as pd
from typing import Dict, Any, Optional
from mlforecast import MLForecast
from lightgbm import LGBMRegressor

class LgbmModel(BaseModel):
    """
    LightGBM wrapper via MLForecast.
    Capabilities:
    - Fast training (Gradient Boosting)
    - Exogenous variables support (native)
    - Autoregressive features (lags)
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.freq = params.get('freq', '1min')
        self.lags = params.get('lags', [1, 12, 24]) # Default lags
        
        self.mlf = MLForecast(
            models=[LGBMRegressor()],
            freq=self.freq,
            lags=self.lags,
        )
        
    def save(self, path: str) -> None:
        print(f"Saving LgbmModel to {path}")
        self.mlf.save(path)

    @classmethod
    def load(cls, path: str) -> 'LgbmModel':
        print(f"Loading LgbmModel from {path}")
        instance = cls()
        instance.mlf = MLForecast.load(path)
        return instance

    def fit(self, df: pd.DataFrame) -> None:
        data = df.copy()
        if 'timestamp' in data.columns: data = data.rename(columns={'timestamp': 'ds'})
        if 'value' in data.columns: data = data.rename(columns={'value': 'y'})
        if 'unique_id' not in data.columns: data['unique_id'] = 'series_0'
        
        self.mlf.fit(data)

    def predict(self, df: pd.DataFrame, horizon: int, confidence_level: Optional[float] = None) -> pd.DataFrame:
        # MLForecast needs 'new_data' (exog) or recursive predict.
        # Simple recursive predict without future exog:
        forecasts = self.mlf.predict(h=horizon)
        
        result = forecasts.reset_index().rename(columns={'ds': 'timestamp'})
        model_name = 'LGBMRegressor'
        
        result['pred'] = result[model_name]
        
        # LGBM point forecast usually, unless QuantileRegressor used.
        result['lower'] = result['pred']
        result['upper'] = result['pred']
            
        return result[['timestamp', 'unique_id', 'pred', 'lower', 'upper']]
