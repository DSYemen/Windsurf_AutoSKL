from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from .base import BaseMLModel

class RegressionModel(BaseMLModel):
    def __init__(self, model_name: str, user_id: int, algorithm: str, params: Dict[str, Any] = None):
        super().__init__(model_name, user_id)
        self.algorithm = algorithm
        self.params = params or {}
        self.model = self._create_model()
        
    def _create_model(self):
        if self.algorithm == "RandomForest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**self.params)
        elif self.algorithm == "XGBoost":
            import xgboost as xgb
            return xgb.XGBRegressor(**self.params)
        elif self.algorithm == "LightGBM":
            import lightgbm as lgb
            return lgb.LGBMRegressor(**self.params)
        # Add more algorithms as needed
        
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # Handle missing values
        data = data.fillna(data.mean())
        
        # Handle categorical variables
        cat_columns = data.select_dtypes(include=['object']).columns
        for col in cat_columns:
            data[col] = pd.Categorical(data[col]).codes
            
        return data
        
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        X_processed = self.preprocess_data(X)
        self.model.fit(X_processed, y)
        
        # Store metadata
        self.metadata.update({
            'features': list(X.columns),
            'target': y.name,
            'training_shape': X.shape,
            'algorithm': self.algorithm,
            'parameters': self.params,
            'cross_val_scores': list(cross_val_score(self.model, X_processed, y, cv=5))
        })
        
    def predict(self, X: pd.DataFrame):
        X_processed = self.preprocess_data(X)
        return self.model.predict(X_processed)
        
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        X_processed = self.preprocess_data(X)
        y_pred = self.predict(X_processed)
        
        return {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
        
    def get_feature_importance(self) -> Dict[str, float]:
        if not hasattr(self.model, 'feature_importances_'):
            return {}
            
        features = self.metadata.get('features', [])
        importances = self.model.feature_importances_
        return dict(zip(features, importances))