from typing import Dict, List, Optional, Union, Any
import time
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from prometheus_client import Counter, Histogram, Gauge
from src.models.ml_model import MLModel
from src.models.prediction import PredictionLog

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    'model_predictions_total',
    'Total number of predictions',
    ['model_id', 'model_version', 'status']
)

PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds',
    'Time spent processing prediction',
    ['model_id', 'model_version']
)

PREDICTION_FEATURE_DRIFT = Gauge(
    'model_feature_drift',
    'Feature drift score',
    ['model_id', 'model_version', 'feature']
)

class ModelMonitor:
    def __init__(self, db_session: Session):
        self.db = db_session
        self.feature_statistics = {}
        
    def log_prediction(
        self,
        model_id: int,
        version: str,
        input_data: Dict[str, Any],
        prediction: Any,
        latency: float,
        status: str = 'success'
    ) -> None:
        """Log prediction details"""
        # Create prediction log
        log_entry = PredictionLog(
            model_id=model_id,
            version=version,
            input_data=json.dumps(input_data),
            prediction=json.dumps(prediction),
            latency=latency,
            status=status,
            timestamp=datetime.utcnow()
        )
        
        # Save to database
        self.db.add(log_entry)
        self.db.commit()
        
        # Update Prometheus metrics
        PREDICTION_COUNTER.labels(
            model_id=model_id,
            model_version=version,
            status=status
        ).inc()
        
        PREDICTION_LATENCY.labels(
            model_id=model_id,
            model_version=version
        ).observe(latency)
        
    def calculate_feature_drift(
        self,
        model_id: int,
        version: str,
        current_data: pd.DataFrame,
        window_days: int = 30
    ) -> Dict[str, float]:
        """Calculate feature drift using statistical methods"""
        # Get historical predictions
        start_date = datetime.utcnow() - timedelta(days=window_days)
        historical_logs = self.db.query(PredictionLog).filter(
            PredictionLog.model_id == model_id,
            PredictionLog.version == version,
            PredictionLog.timestamp >= start_date
        ).all()
        
        if not historical_logs:
            return {}
            
        # Convert historical data to DataFrame
        historical_data = pd.DataFrame([
            json.loads(log.input_data)
            for log in historical_logs
        ])
        
        drift_scores = {}
        
        for feature in current_data.columns:
            if feature not in historical_data.columns:
                continue
                
            # Calculate drift score using Kolmogorov-Smirnov test
            from scipy import stats
            ks_statistic, _ = stats.ks_2samp(
                historical_data[feature],
                current_data[feature]
            )
            
            drift_scores[feature] = ks_statistic
            
            # Update Prometheus metric
            PREDICTION_FEATURE_DRIFT.labels(
                model_id=model_id,
                model_version=version,
                feature=feature
            ).set(ks_statistic)
            
        return drift_scores
        
    def get_performance_metrics(
        self,
        model_id: int,
        version: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get model performance metrics"""
        query = self.db.query(PredictionLog).filter(
            PredictionLog.model_id == model_id,
            PredictionLog.version == version
        )
        
        if start_date:
            query = query.filter(PredictionLog.timestamp >= start_date)
        if end_date:
            query = query.filter(PredictionLog.timestamp <= end_date)
            
        logs = query.all()
        
        if not logs:
            return {}
            
        # Calculate metrics
        total_predictions = len(logs)
        success_predictions = len([log for log in logs if log.status == 'success'])
        error_predictions = total_predictions - success_predictions
        
        latencies = [log.latency for log in logs]
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        return {
            'total_predictions': total_predictions,
            'success_rate': success_predictions / total_predictions,
            'error_rate': error_predictions / total_predictions,
            'latency': {
                'average': avg_latency,
                'p95': p95_latency,
                'p99': p99_latency
            }
        }
        
    def detect_anomalies(
        self,
        model_id: int,
        version: str,
        current_data: pd.DataFrame,
        threshold: float = 3.0
    ) -> Dict[str, List[str]]:
        """Detect anomalies in input features"""
        # Get or calculate feature statistics
        stats_key = f"{model_id}_{version}"
        if stats_key not in self.feature_statistics:
            # Get historical data
            historical_logs = self.db.query(PredictionLog).filter(
                PredictionLog.model_id == model_id,
                PredictionLog.version == version
            ).all()
            
            if not historical_logs:
                return {}
                
            historical_data = pd.DataFrame([
                json.loads(log.input_data)
                for log in historical_logs
            ])
            
            # Calculate statistics
            self.feature_statistics[stats_key] = {
                feature: {
                    'mean': historical_data[feature].mean(),
                    'std': historical_data[feature].std()
                }
                for feature in historical_data.columns
                if historical_data[feature].dtype in ['int64', 'float64']
            }
            
        # Detect anomalies
        anomalies = {}
        for feature, stats in self.feature_statistics[stats_key].items():
            if feature not in current_data.columns:
                continue
                
            # Calculate z-scores
            z_scores = np.abs(
                (current_data[feature] - stats['mean']) / stats['std']
            )
            
            # Find anomalies
            anomaly_indices = np.where(z_scores > threshold)[0]
            if len(anomaly_indices) > 0:
                anomalies[feature] = anomaly_indices.tolist()
                
        return anomalies
        
    def generate_monitoring_report(
        self,
        model_id: int,
        version: str,
        window_days: int = 30
    ) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=window_days)
        
        # Get performance metrics
        performance = self.get_performance_metrics(
            model_id,
            version,
            start_date,
            end_date
        )
        
        # Get recent predictions
        recent_logs = self.db.query(PredictionLog).filter(
            PredictionLog.model_id == model_id,
            PredictionLog.version == version,
            PredictionLog.timestamp >= start_date
        ).all()
        
        if not recent_logs:
            return {'performance': performance}
            
        recent_data = pd.DataFrame([
            json.loads(log.input_data)
            for log in recent_logs
        ])
        
        # Calculate drift
        drift_scores = self.calculate_feature_drift(
            model_id,
            version,
            recent_data
        )
        
        # Detect anomalies
        anomalies = self.detect_anomalies(
            model_id,
            version,
            recent_data
        )
        
        return {
            'performance': performance,
            'feature_drift': drift_scores,
            'anomalies': anomalies,
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }
        }
