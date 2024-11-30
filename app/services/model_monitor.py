from typing import Dict, Any, Optional
import numpy as np
from scipy.stats import ks_2samp
import pandas as pd
from datetime import datetime
import json
import logging
from .data_processor import DataProcessor
from .model_trainer import ModelTrainer

class ModelMonitor:
    def __init__(
        self,
        drift_threshold: float = 0.05,
        performance_threshold: float = 0.1
    ):
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        self.reference_data = None
        self.reference_stats = None
        
    def set_reference_data(self, X: np.ndarray):
        """Set reference data for drift detection"""
        self.reference_data = X
        self.reference_stats = self._calculate_statistics(X)
        
    def _calculate_statistics(self, X: np.ndarray) -> Dict[str, Any]:
        """Calculate basic statistics for data"""
        stats = {
            'mean': np.mean(X, axis=0).tolist(),
            'std': np.std(X, axis=0).tolist(),
            'min': np.min(X, axis=0).tolist(),
            'max': np.max(X, axis=0).tolist()
        }
        return stats
        
    def detect_drift(self, new_data: np.ndarray) -> Dict[str, Any]:
        """Detect if there is significant drift in new data"""
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data first.")
            
        drift_scores = []
        for i in range(new_data.shape[1]):
            statistic, p_value = ks_2samp(
                self.reference_data[:, i],
                new_data[:, i]
            )
            drift_scores.append({
                'feature_index': i,
                'statistic': float(statistic),
                'p_value': float(p_value),
                'has_drift': p_value < self.drift_threshold
            })
            
        return {
            'drift_detected': any(d['has_drift'] for d in drift_scores),
            'feature_drift_scores': drift_scores,
            'timestamp': datetime.now().isoformat()
        }
        
    def monitor_model_performance(
        self,
        model: ModelTrainer,
        X: np.ndarray,
        y: np.ndarray,
        metric_func: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Monitor model performance on new data"""
        predictions = model.predict(X)
        
        if metric_func is None:
            if model.task_type == 'classification':
                from sklearn.metrics import accuracy_score
                metric_func = accuracy_score
            else:
                from sklearn.metrics import r2_score
                metric_func = r2_score
                
        current_score = metric_func(y, predictions)
        performance_drop = model.best_score - current_score
        
        return {
            'current_score': float(current_score),
            'performance_drop': float(performance_drop),
            'requires_retraining': performance_drop > self.performance_threshold,
            'timestamp': datetime.now().isoformat()
        }
        
    def save_monitoring_report(
        self,
        drift_report: Dict[str, Any],
        performance_report: Dict[str, Any],
        path: str
    ):
        """Save monitoring results to a file"""
        report = {
            'drift_analysis': drift_report,
            'performance_analysis': performance_report,
            'reference_statistics': self.reference_stats,
            'monitoring_config': {
                'drift_threshold': self.drift_threshold,
                'performance_threshold': self.performance_threshold
            }
        }
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
            
    def load_monitoring_report(self, path: str) -> Dict[str, Any]:
        """Load monitoring results from a file"""
        with open(path, 'r') as f:
            return json.load(f)
