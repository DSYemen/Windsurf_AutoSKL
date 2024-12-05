from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd
from sklearn.base import BaseEstimator
import joblib
from src.config import settings

class BaseMLModel(ABC):
    def __init__(self, model_name: str, user_id: int):
        self.model_name = model_name
        self.user_id = user_id
        self.model: Optional[BaseEstimator] = None
        self.model_path = settings.MODEL_STORE_PATH / f"user_{user_id}" / f"{model_name}.joblib"
        self.metadata: Dict[str, Any] = {}
        
    @abstractmethod
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
        
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        pass
        
    @abstractmethod
    def predict(self, X: pd.DataFrame):
        pass
        
    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        pass
        
    def save_model(self):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        model_data = {
            'model': self.model,
            'metadata': self.metadata
        }
        joblib.dump(model_data, self.model_path)
        
    def load_model(self):
        if self.model_path.exists():
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.metadata = model_data['metadata']
            return True
        return False
