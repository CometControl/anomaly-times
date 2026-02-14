from ..base import BaseModel
import pandas as pd
from typing import Dict, Any, Optional
from prefect import flow
from ..utils import run_stateful_model
try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA
    STATSFORECAST_AVAILABLE = True
except ImportError:
    STATSFORECAST_AVAILABLE = False

class MockStatsForecast:
    def __init__(self, *args, **kwargs):
        pass
        
    def fit(self, df):
        self.last_df = df
        
    def predict(self, h, level=None):
        # Generate dummy forecast
        import numpy as np
        
        unique_ids = self.last_df['unique_id'].unique()
        last_date = self.last_df['ds'].max()
        
        start_date = last_date - pd.Timedelta(minutes=h) # Backcast for overlap testing
        dates = pd.date_range(start=start_date, periods=h, freq='1min')
        
        rows = []
        for uid in unique_ids:
            for dt in dates:
                # Random value around 0
                val = np.random.randn()
                row = {
                    'unique_id': uid,
                    'ds': dt,
                    'AutoARIMA': val,
                }
                if level:
                    for l in level:
                        row[f'AutoARIMA-lo-{l}'] = val - 1.0
                        row[f'AutoARIMA-hi-{l}'] = val + 1.0
                rows.append(row)
                
        return pd.DataFrame(rows).set_index('unique_id')
        
    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, path):
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)

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
        
        if STATSFORECAST_AVAILABLE:
            # Parallel Univariate: StatsForecast fits independent ARIMA per series
            self.sf = StatsForecast(
                models=[AutoARIMA(season_length=self.season_length)],
                freq=self.freq,
                n_jobs=self.n_jobs
            )
        else:
            print("Warning: statsforecast not found. Using MockStatsForecast.")
            self.sf = MockStatsForecast()
        
    def save(self, path: str) -> None:
        # StatsForecast has a save method, but it saves the whole object
        print(f"Saving ArimaModel to {path}")
        self.sf.save(path=path)

    @classmethod
    def load(cls, path: str) -> 'ArimaModel':
        print(f"Loading ArimaModel from {path}")
        # Create empty instance
        instance = cls()
        
        if STATSFORECAST_AVAILABLE:
            try:
                # Load internal StatsForecast object
                instance.sf = StatsForecast.load(path=path)
            except ImportError:
                 # Fallback if somehow STATSFORECAST_AVAILABLE is True but load fails? 
                 # Unlikely, but let's be safe or just re-raise.
                 # Actually, if we are loading a pickled MockStatsForecast while STATSFORECAST_AVAILABLE is True, 
                 # StatsForecast.load might fail or work depending on pickle.
                 # But sticking to simple logic:
                 instance.sf = StatsForecast.load(path=path)
        else:
            print("Warning: statsforecast not found. Using MockStatsForecast load.")
            instance.sf = MockStatsForecast.load(path=path)
            
        return instance

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

@flow(name="arima_subflow")
def arima_flow(
    context_df: pd.DataFrame,
    horizon: int = 60,
    confidence_level: float = 0.9,
    storage_path: Optional[str] = None,
    fit_expiration_hours: int = 24,
    **kwargs
) -> pd.DataFrame:
    
    params = {
        "freq": "1min", # Could be parameterized further if needed
    }
    
    return run_stateful_model(
        model_class=ArimaModel,
        context_df=context_df,
        params=params,
        horizon=horizon,
        confidence_level=confidence_level,
        storage_path=storage_path,
        fit_expiration_hours=fit_expiration_hours
    )
