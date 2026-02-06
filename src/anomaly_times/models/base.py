from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Tuple, Dict, Any

class BaseModel(ABC):
    """
    Abstract base class for all forecasting models in Anomaly Times.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """
        Fits the model on the provided historical dataframe.
        
        Args:
            df: DataFrame with 'timestamp', 'value', and optional label columns.
                Must be compatible with the specific library (Nixtla/Chronos) expected format.
        """
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame, horizon: int, confidence_level: Optional[float] = None) -> pd.DataFrame:
        """
        Generates forecast for the specified horizon.
        
        Args:
            df: DataFrame with context/recent history.
            horizon: Number of steps to forecast.
            confidence_level: Optional confidence level (e.g., 0.9 for 90% interval).
            
        Returns:
            pd.DataFrame: DataFrame containing:
                - 'timestamp': Future timestamps
                - 'pred': Point forecast
                - 'lower': Lower bound (if confidence_level set)
                - 'upper': Upper bound (if confidence_level set)
        """
        pass
