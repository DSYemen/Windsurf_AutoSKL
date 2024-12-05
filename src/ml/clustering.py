from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from .base import BaseMLModel

class ClusteringModel(BaseMLModel):
    def __init__(self, model_name: str, user_id: int, algorithm: str, params: Dict[str, Any] = None):
        super().__init__(model_name, user_id)
        self.algorithm = algorithm
        self.params = params or {}
        self.model = self._create_model()
        
    def _create_model(self):
        if self.algorithm == "KMeans":
            return KMeans(**self.params)
        elif self.algorithm == "DBSCAN":
            return DBSCAN(**self.params)
        elif self.algorithm == "Agglomerative":
            return AgglomerativeClustering(**self.params)
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
        self.model.fit(X_processed)
        
        # Calculate clustering metrics
        labels = self.model.labels_ if hasattr(self.model, 'labels_') else self.model.fit_predict(X_processed)
        
        metrics = {
            'silhouette_score': silhouette_score(X_processed, labels),
            'calinski_harabasz_score': calinski_harabasz_score(X_processed, labels),
            'davies_bouldin_score': davies_bouldin_score(X_processed, labels),
            'n_clusters': len(np.unique(labels))
        }
        
        # Store metadata
        self.metadata.update({
            'features': list(X.columns),
            'training_shape': X.shape,
            'algorithm': self.algorithm,
            'parameters': self.params,
            'metrics': metrics
        })
        
    def predict(self, X: pd.DataFrame):
        X_processed = self.preprocess_data(X)
        if hasattr(self.model, 'predict'):
            return self.model.predict(X_processed)
        return self.model.fit_predict(X_processed)
        
    def evaluate(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, float]:
        X_processed = self.preprocess_data(X)
        labels = self.predict(X_processed)
        
        return {
            'silhouette_score': silhouette_score(X_processed, labels),
            'calinski_harabasz_score': calinski_harabasz_score(X_processed, labels),
            'davies_bouldin_score': davies_bouldin_score(X_processed, labels),
            'n_clusters': len(np.unique(labels))
        }
        
    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """Get cluster centers if available"""
        if hasattr(self.model, 'cluster_centers_'):
            return self.model.cluster_centers_
        return None
        
    def get_cluster_distribution(self, X: pd.DataFrame) -> Dict[str, int]:
        """Get distribution of samples across clusters"""
        labels = self.predict(X)
        unique_labels, counts = np.unique(labels, return_counts=True)
        return {f"cluster_{label}": count for label, count in zip(unique_labels, counts)}
