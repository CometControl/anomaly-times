from ..base import BaseModel
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import TimesNet
from prefect import flow
from ..utils import run_stateful_model

class TimesNetModel(BaseModel):
    """
    TimesNet wrapper via NeuralForecast.
    Capabilities:
    - Multi-periodicity analysis (2D variations)
    - Global model
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.freq = params.get('freq', '1min')
        self.input_size = params.get('input_size', 60)
        self.h = params.get('horizon', 60)
        self.max_steps = params.get('max_steps', 100)
        
        models = [TimesNet(
            h=self.h,
            input_size=self.input_size,
            max_steps=self.max_steps,
            scaler_type='standard'
        )]
        
        self.nf = NeuralForecast(
            models=models,
            freq=self.freq
        )
        
    def save(self, path: str) -> None:
        print(f"Saving TimesNetModel to {path}")
        # NeuralForecast saving creates a directory at path
        self.nf.save(path=path, overwrite=True)

    @classmethod
    def load(cls, path: str) -> 'TimesNetModel':
        print(f"Loading TimesNetModel from {path}")
        instance = cls()
        instance.nf = NeuralForecast.load(path=path)
        return instance

    def fit(self, df: pd.DataFrame) -> None:
        data = df.copy()
        if 'timestamp' in data.columns: data = data.rename(columns={'timestamp': 'ds'})
        if 'value' in data.columns: data = data.rename(columns={'value': 'y'})
        if 'unique_id' not in data.columns: data['unique_id'] = 'series_0'
        
        self.nf.fit(df=data)

    def predict(self, df: pd.DataFrame, horizon: int, confidence_level: Optional[float] = None) -> pd.DataFrame:
        forecasts = self.nf.predict() 
        
        result = forecasts.reset_index().rename(columns={'ds': 'timestamp'})
        model_name = 'TimesNet'
        
        result['pred'] = result[model_name]
        
        # Basic confidence fallback
        result['lower'] = result['pred']
        result['upper'] = result['pred']
            
        return result[['timestamp', 'unique_id', 'pred', 'lower', 'upper']]

@flow(name="timesnet_subflow")
def timesnet_flow(
    context_df: pd.DataFrame,
    horizon: int = 60,
    confidence_level: float = 0.9,
    storage_path: str = None,
    fit_expiration_hours: int = 24,
    multivariate: bool = False,
    **kwargs
) -> pd.DataFrame:
    
    params = {
        "freq": "1min",
        "horizon": horizon,
        "multivariate": multivariate
    }
    
    return run_stateful_model(
        model_class=TimesNetModel,
        context_df=context_df,
        params=params,
        horizon=horizon,
        confidence_level=confidence_level,
        storage_path=storage_path,
        fit_expiration_hours=fit_expiration_hours
    )
