import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataProcessor:
    def __init__(self):
        self.numerical_pipeline = None
        self.categorical_pipeline = None
        self.target_encoder = None
        self.feature_names = None
        
    def _identify_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        return numerical_columns, categorical_columns
    
    def fit_transform(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fit the preprocessor and transform the data"""
        numerical_columns, categorical_columns = self._identify_column_types(df)
        
        # Remove target column from features if specified
        if target_column:
            if target_column in numerical_columns:
                numerical_columns.remove(target_column)
            if target_column in categorical_columns:
                categorical_columns.remove(target_column)
        
        # Create preprocessing pipelines
        self.numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        self.categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', LabelEncoder())
        ])
        
        # Create column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numerical_pipeline, numerical_columns),
                ('cat', self.categorical_pipeline, categorical_columns)
            ])
        
        # Transform features
        X = preprocessor.fit_transform(df.drop(columns=[target_column] if target_column else []))
        self.feature_names = numerical_columns + categorical_columns
        
        # Transform target if specified
        y = None
        if target_column:
            self.target_encoder = LabelEncoder()
            y = self.target_encoder.fit_transform(df[target_column])
        
        return X, y
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessor"""
        if self.numerical_pipeline is None or self.categorical_pipeline is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
            
        numerical_columns, categorical_columns = self._identify_column_types(df)
        
        # Transform features
        X = df[self.feature_names].copy()
        for col in numerical_columns:
            X[col] = self.numerical_pipeline.transform(X[[col]])
        for col in categorical_columns:
            X[col] = self.categorical_pipeline.transform(X[[col]])
            
        return X.values
