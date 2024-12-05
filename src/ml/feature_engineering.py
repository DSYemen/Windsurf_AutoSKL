from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from category_encoders import TargetEncoder, WOEEncoder, CatBoostEncoder
import featuretools as ft
from featuretools.primitives import (
    Sum, Mean, Max, Min, Std,
    Month, Weekday, Hour,
    NumCharacters, NumWords
)

class FeatureEngineer:
    def __init__(
        self,
        task_type: str,
        target_column: str,
        feature_selection: bool = True,
        max_features: Optional[int] = None
    ):
        self.task_type = task_type
        self.target_column = target_column
        self.feature_selection = feature_selection
        self.max_features = max_features
        self.feature_scores = None
        self.selected_features = None
        self.categorical_encoders = {}
        
    def _create_date_features(self, df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
        """Create features from datetime columns"""
        result = df.copy()
        
        for col in date_columns:
            if not pd.api.types.is_datetime64_any_dtype(result[col]):
                result[col] = pd.to_datetime(result[col], errors='ignore')
                
            if pd.api.types.is_datetime64_any_dtype(result[col]):
                result[f'{col}_year'] = result[col].dt.year
                result[f'{col}_month'] = result[col].dt.month
                result[f'{col}_day'] = result[col].dt.day
                result[f'{col}_weekday'] = result[col].dt.weekday
                result[f'{col}_hour'] = result[col].dt.hour
                result[f'{col}_is_weekend'] = result[col].dt.weekday.isin([5, 6]).astype(int)
                
        return result
        
    def _create_text_features(self, df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
        """Create features from text columns"""
        result = df.copy()
        
        for col in text_columns:
            if df[col].dtype == 'object':
                result[f'{col}_length'] = df[col].str.len()
                result[f'{col}_word_count'] = df[col].str.split().str.len()
                result[f'{col}_unique_chars'] = df[col].str.nunique()
                result[f'{col}_uppercase_ratio'] = (
                    df[col].str.count(r'[A-Z]') / df[col].str.len()
                )
                
        return result
        
    def _create_numeric_interactions(
        self,
        df: pd.DataFrame,
        numeric_columns: List[str],
        degree: int = 2
    ) -> pd.DataFrame:
        """Create interaction features for numeric columns"""
        if len(numeric_columns) < 2:
            return df
            
        result = df.copy()
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        
        # Create interactions for subsets of columns to avoid explosion
        for i in range(0, len(numeric_columns), 3):
            subset = numeric_columns[i:i+3]
            if len(subset) > 1:
                interactions = poly.fit_transform(df[subset])
                feature_names = poly.get_feature_names_out(subset)
                
                # Add only interaction terms (skip original features)
                for j, name in enumerate(feature_names[len(subset):], start=len(subset)):
                    result[f'interaction_{name}'] = interactions[:, j]
                    
        return result
        
    def _create_aggregation_features(
        self,
        df: pd.DataFrame,
        group_columns: List[str],
        agg_columns: List[str]
    ) -> pd.DataFrame:
        """Create aggregation features based on groupby operations"""
        result = df.copy()
        
        for group_col in group_columns:
            for agg_col in agg_columns:
                # Skip if either column doesn't exist
                if group_col not in df.columns or agg_col not in df.columns:
                    continue
                    
                # Calculate aggregations
                aggs = df.groupby(group_col)[agg_col].agg([
                    'mean', 'std', 'min', 'max', 'count'
                ]).add_prefix(f'{agg_col}_by_{group_col}_')
                
                # Merge back to original dataframe
                result = result.merge(
                    aggs,
                    left_on=group_col,
                    right_index=True,
                    how='left'
                )
                
        return result
        
    def _encode_categorical_features(
        self,
        df: pd.DataFrame,
        categorical_columns: List[str],
        target: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Encode categorical features using various encoding strategies"""
        result = df.copy()
        
        for col in categorical_columns:
            if col not in self.categorical_encoders:
                if target is not None and self.task_type in ['classification', 'regression']:
                    # Use target encoding for supervised learning
                    self.categorical_encoders[col] = TargetEncoder()
                    result[f'{col}_encoded'] = self.categorical_encoders[col].fit_transform(
                        result[col],
                        target
                    )
                else:
                    # Use frequency encoding for unsupervised learning
                    value_counts = df[col].value_counts(normalize=True)
                    self.categorical_encoders[col] = value_counts
                    result[f'{col}_encoded'] = result[col].map(value_counts)
            else:
                if isinstance(self.categorical_encoders[col], TargetEncoder):
                    result[f'{col}_encoded'] = self.categorical_encoders[col].transform(result[col])
                else:
                    result[f'{col}_encoded'] = result[col].map(self.categorical_encoders[col])
                    
        return result
        
    def _select_features(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Select most important features"""
        if not self.feature_selection or y is None:
            return X, {col: 1.0 for col in X.columns}
            
        # Choose appropriate scoring function
        if self.task_type == 'classification':
            score_func = mutual_info_classif
        else:
            score_func = mutual_info_regression
            
        # Calculate feature scores
        k = self.max_features if self.max_features else 'all'
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)
        
        # Get feature scores
        feature_scores = dict(zip(X.columns, selector.scores_))
        
        # Select features
        selected_cols = X.columns[selector.get_support()].tolist()
        
        return X[selected_cols], feature_scores
        
    def fit_transform(
        self,
        df: pd.DataFrame,
        date_columns: Optional[List[str]] = None,
        text_columns: Optional[List[str]] = None,
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        group_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create and select features"""
        result = df.copy()
        
        # Identify column types if not provided
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
        if categorical_columns is None:
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
        if date_columns is None:
            date_columns = [
                col for col in df.columns
                if pd.api.types.is_datetime64_any_dtype(df[col])
            ]
            
        # Create features
        if date_columns:
            result = self._create_date_features(result, date_columns)
            
        if text_columns:
            result = self._create_text_features(result, text_columns)
            
        if numeric_columns:
            result = self._create_numeric_interactions(result, numeric_columns)
            
        if categorical_columns:
            target = df[self.target_column] if self.target_column in df else None
            result = self._encode_categorical_features(result, categorical_columns, target)
            
        if group_columns and numeric_columns:
            result = self._create_aggregation_features(result, group_columns, numeric_columns)
            
        # Select features
        if self.feature_selection and self.target_column in df:
            target = df[self.target_column]
            features = result.drop(columns=[self.target_column])
            selected_features, self.feature_scores = self._select_features(features, target)
            result = pd.concat([selected_features, target], axis=1)
            
        # Store metadata
        metadata = {
            'original_shape': df.shape,
            'transformed_shape': result.shape,
            'created_features': list(set(result.columns) - set(df.columns)),
            'feature_scores': self.feature_scores,
            'categorical_encodings': {
                col: type(encoder).__name__
                for col, encoder in self.categorical_encoders.items()
            }
        }
        
        return result, metadata
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted encoders"""
        result = df.copy()
        
        # Apply categorical encodings
        for col, encoder in self.categorical_encoders.items():
            if col in result.columns:
                if isinstance(encoder, TargetEncoder):
                    result[f'{col}_encoded'] = encoder.transform(result[col])
                else:
                    result[f'{col}_encoded'] = result[col].map(encoder)
                    
        return result
