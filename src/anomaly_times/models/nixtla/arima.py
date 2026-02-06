from ..base import BaseModel
import pandas as pd
from typing import Dict, Any, Optional
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

class ArimaModel(BaseModel):
    """
    AutoARIMA wrapper via StatsForecast.
    Type: Parallel Univariate.
    Capabilities:
    - seasonality
    - confidence intervals
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.freq = params.get('freq', '1min')
        self.season_length = params.get('season_length', 60)
        self.n_jobs = params.get('n_jobs', -1)
        
        # Parallel Univariate: StatsForecast fits independent ARIMA per series
        self.sf = StatsForecast(
            models=[AutoARIMA(season_length=self.season_length)],
            freq=self.freq,
            n_jobs=self.n_jobs
        )

    def fit(self, df: pd.DataFrame) -> None:
        data = df.copy()
        if 'timestamp' in data.columns: data = data.rename(columns={'timestamp': 'ds'})
        if 'value' in data.columns: data = data.rename(columns={'value': 'y'})
        if 'unique_id' not in data.columns: data['unique_id'] = 'series_0'
        
        self.sf.fit(data)

    def predict(self, df: pd.DataFrame, horizon: int, confidence_level: Optional[float] = None) -> pd.DataFrame:
        levels = [int(confidence_level * 100)] if confidence_level else []
        
        forecasts = self.sf.predict(h=horizon, level=levels)
        
        result = forecasts.reset_index().rename(columns={'ds': 'timestamp'})
        model_name = 'AutoARIMA'
        
        result['pred'] = result[model_name]
        
        if confidence_level:
            lvl = int(confidence_level * 100)
            result['lower'] = result.get(f"{model_name}-lo-{lvl}", result['pred'])
            result['upper'] = result.get(f"{model_name}-hi-{lvl}", result['pred'])
        else:
            result['lower'] = result['pred']
            result['upper'] = result['pred']
            
        return result[['timestamp', 'unique_id', 'pred', 'lower', 'upper']]
