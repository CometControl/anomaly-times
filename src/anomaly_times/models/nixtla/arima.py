from ..base import BaseModel
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from prefect import flow
from ..utils import run_stateful_model

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
        
    def save(self, path: str) -> None:
        # StatsForecast has a save method, but it saves the whole object
        print(f"Saving ArimaModel to {path}")
        self.sf.save(path=path)

    @classmethod
    def load(cls, path: str) -> 'ArimaModel':
        print(f"Loading ArimaModel from {path}")
        # Create empty instance
        instance = cls()
        # Load internal StatsForecast object
        instance.sf = StatsForecast.load(path=path)
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
    storage_path: str = None,
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
