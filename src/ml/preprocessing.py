from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder, BinaryEncoder
import json

class DataPreprocessor:
    def __init__(self):
        self.categorical_encoders: Dict[str, Union[LabelEncoder, TargetEncoder, BinaryEncoder]] = {}
        self.numerical_scalers: Dict[str, Union[StandardScaler, MinMaxScaler]] = {}
        self.imputers: Dict[str, SimpleImputer] = {}
        self.feature_types: Dict[str, str] = {}
        self.metadata: Dict[str, Any] = {}

    def analyze_features(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze features and their characteristics"""
        analysis = {}
        for column in df.columns:
            stats = {
                'dtype': str(df[column].dtype),
                'missing_count': df[column].isnull().sum(),
                'unique_count': df[column].nunique(),
                'memory_usage': df[column].memory_usage(deep=True),
            }
            
            if pd.api.types.is_numeric_dtype(df[column]):
                stats.update({
                    'mean': df[column].mean(),
                    'std': df[column].std(),
                    'min': df[column].min(),
                    'max': df[column].max(),
                    'type': 'numeric'
                })
            else:
                stats.update({
                    'most_common': df[column].value_counts().head(5).to_dict(),
                    'type': 'categorical'
                })
            
            analysis[column] = stats
        return analysis

    def fit_transform(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """Fit preprocessors and transform data"""
        self.metadata['original_shape'] = df.shape
        self.metadata['feature_analysis'] = self.analyze_features(df)
        
        # Create copy to avoid modifying original
        df_processed = df.copy()
        
        # Handle missing values first
        for column in df_processed.columns:
            if df_processed[column].isnull().any():
                if pd.api.types.is_numeric_dtype(df_processed[column]):
                    self.imputers[column] = SimpleImputer(strategy='mean')
                else:
                    self.imputers[column] = SimpleImputer(strategy='most_frequent')
                df_processed[column] = self.imputers[column].fit_transform(df_processed[[column]])
        
        # Process features
        for column in df_processed.columns:
            if column == target_column:
                continue
                
            if pd.api.types.is_numeric_dtype(df_processed[column]):
                self.feature_types[column] = 'numeric'
                self.numerical_scalers[column] = StandardScaler()
                df_processed[column] = self.numerical_scalers[column].fit_transform(df_processed[[column]])
            else:
                self.feature_types[column] = 'categorical'
                if df_processed[column].nunique() == 2:
                    self.categorical_encoders[column] = BinaryEncoder()
                elif target_column and df_processed[column].nunique() > 10:
                    self.categorical_encoders[column] = TargetEncoder()
                    df_processed[column] = self.categorical_encoders[column].fit_transform(
                        df_processed[[column]], 
                        df_processed[target_column]
                    )
                else:
                    self.categorical_encoders[column] = LabelEncoder()
                    df_processed[column] = self.categorical_encoders[column].fit_transform(df_processed[column])
        
        self.metadata['processed_shape'] = df_processed.shape
        return df_processed, self.metadata

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessors"""
        df_processed = df.copy()
        
        # Apply imputation
        for column, imputer in self.imputers.items():
            if column in df_processed.columns:
                df_processed[column] = imputer.transform(df_processed[[column]])
        
        # Apply transformations
        for column in df_processed.columns:
            if column in self.numerical_scalers:
                df_processed[column] = self.numerical_scalers[column].transform(df_processed[[column]])
            elif column in self.categorical_encoders:
                if isinstance(self.categorical_encoders[column], TargetEncoder):
                    df_processed[column] = self.categorical_encoders[column].transform(df_processed[[column]])
                else:
                    df_processed[column] = self.categorical_encoders[column].transform(df_processed[column])
        
        return df_processed

    def save_state(self, path: str):
        """Save preprocessor state"""
        state = {
            'metadata': self.metadata,
            'feature_types': self.feature_types
        }
        with open(path, 'w') as f:
            json.dump(state, f)

    def load_state(self, path: str):
        """Load preprocessor state"""
        with open(path, 'r') as f:
            state = json.load(f)
        self.metadata = state['metadata']
        self.feature_types = state['feature_types']
