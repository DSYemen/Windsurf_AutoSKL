from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.manifold import TSNE, UMAP
from .base import BaseMLModel

class DimensionalityReductionModel(BaseMLModel):
    def __init__(self, model_name: str, user_id: int, algorithm: str, params: Dict[str, Any] = None):
        super().__init__(model_name, user_id)
        self.algorithm = algorithm
        self.params = params or {}
        self.model = self._create_model()
        self.feature_names_out = None
        
    def _create_model(self):
        if self.algorithm == "PCA":
            return PCA(**self.params)
        elif self.algorithm == "ICA":
            return FastICA(**self.params)
        elif self.algorithm == "NMF":
            return NMF(**self.params)
        elif self.algorithm == "TSNE":
            return TSNE(**self.params)
        elif self.algorithm == "UMAP":
            from umap import UMAP as UMAP_MODEL
            return UMAP_MODEL(**self.params)
        raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # Handle missing values
        data = data.fillna(data.mean())
        
        # Handle categorical variables
        cat_columns = data.select_dtypes(include=['object']).columns
        for col in cat_columns:
            data[col] = pd.Categorical(data[col]).codes
            
        return data
        
    def train(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs):
        X_processed = self.preprocess_data(X)
        transformed_data = self.model.fit_transform(X_processed)
        
        # Generate feature names for transformed data
        self.feature_names_out = [f"component_{i+1}" for i in range(transformed_data.shape[1])]
        
        # Calculate explained variance if available
        explained_variance = None
        if hasattr(self.model, 'explained_variance_ratio_'):
            explained_variance = self.model.explained_variance_ratio_.tolist()
        
        # Store metadata
        self.metadata.update({
            'original_features': list(X.columns),
            'transformed_features': self.feature_names_out,
            'original_shape': X.shape,
            'transformed_shape': transformed_data.shape,
            'algorithm': self.algorithm,
            'parameters': self.params,
            'explained_variance': explained_variance
        })
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data to reduced dimensions"""
        X_processed = self.preprocess_data(X)
        transformed_data = self.model.transform(X_processed)
        return pd.DataFrame(
            transformed_data,
            columns=self.feature_names_out,
            index=X.index
        )
        
    def inverse_transform(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Reconstruct original data from transformed data if possible"""
        if hasattr(self.model, 'inverse_transform'):
            reconstructed = self.model.inverse_transform(X)
            return pd.DataFrame(
                reconstructed,
                columns=self.metadata['original_features'],
                index=X.index
            )
        return None
        
    def get_components(self) -> Optional[pd.DataFrame]:
        """Get component matrix if available"""
        if hasattr(self.model, 'components_'):
            return pd.DataFrame(
                self.model.components_,
                columns=self.metadata['original_features'],
                index=self.feature_names_out
            )
        return None
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on components"""
        components = self.get_components()
        if components is not None:
            # Calculate feature importance as the sum of absolute values of components
            importance = np.abs(components.values).sum(axis=0)
            return dict(zip(self.metadata['original_features'], importance))
        return {}
        
    def evaluate(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, float]:
        """Evaluate the dimensionality reduction"""
        metrics = {
            'n_components': len(self.feature_names_out),
            'compression_ratio': X.shape[1] / len(self.feature_names_out)
        }
        
        # Add explained variance if available
        if self.metadata.get('explained_variance'):
            metrics['total_explained_variance'] = sum(self.metadata['explained_variance'])
            
        # Add reconstruction error if possible
        if hasattr(self.model, 'inverse_transform'):
            X_processed = self.preprocess_data(X)
            transformed = self.model.transform(X_processed)
            reconstructed = self.model.inverse_transform(transformed)
            reconstruction_error = np.mean((X_processed - reconstructed) ** 2)
            metrics['reconstruction_error'] = reconstruction_error
            
        return metrics
        
    def predict(self, X: pd.DataFrame):
        """Alias for transform to maintain compatibility with base class"""
        return self.transform(X)
