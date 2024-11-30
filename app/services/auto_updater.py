from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging
from celery import Celery
from .model_trainer import ModelTrainer
from .model_monitor import ModelMonitor
from .data_processor import DataProcessor
from .report_generator import ReportGenerator

# Configure Celery
celery_app = Celery('auto_updater',
                    broker='redis://localhost:6379/0',
                    backend='redis://localhost:6379/0')

class AutoUpdater:
    def __init__(
        self,
        model_dir: str = "models",
        update_threshold: float = 0.1,
        min_samples_required: int = 1000,
        update_frequency: timedelta = timedelta(days=1)
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.update_threshold = update_threshold
        self.min_samples_required = min_samples_required
        self.update_frequency = update_frequency
        
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer()
        self.model_monitor = ModelMonitor()
        self.report_generator = ReportGenerator()
        
        # Load existing model if available
        model_path = self.model_dir / "model.joblib"
        if model_path.exists():
            self.model_trainer = ModelTrainer.load_model(str(model_path))
            
    def _load_metadata(self) -> Dict[str, Any]:
        """Load model metadata"""
        metadata_path = self.model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {
            'last_update': None,
            'version': 0,
            'performance_history': []
        }
        
    def _save_metadata(self, metadata: Dict[str, Any]):
        """Save model metadata"""
        metadata_path = self.model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def collect_new_data(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Collect and analyze new data"""
        # Detect drift
        drift_report = self.model_monitor.detect_drift(X)
        
        # If labels are available, check performance
        performance_report = None
        if y is not None:
            performance_report = self.model_monitor.monitor_model_performance(
                self.model_trainer, X, y
            )
            
        # Generate monitoring report
        monitoring_path = self.model_dir / "monitoring.json"
        report = {
            'timestamp': datetime.now().isoformat(),
            'drift_analysis': drift_report,
            'performance_analysis': performance_report,
            'data_size': len(X)
        }
        
        with open(monitoring_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        return report
        
    def should_update(self, monitoring_report: Dict[str, Any]) -> bool:
        """Determine if model should be updated"""
        metadata = self._load_metadata()
        
        # Check if enough time has passed since last update
        if metadata['last_update']:
            last_update = datetime.fromisoformat(metadata['last_update'])
            if datetime.now() - last_update < self.update_frequency:
                return False
                
        # Check if we have enough new data
        if monitoring_report['data_size'] < self.min_samples_required:
            return False
            
        # Check for significant drift or performance degradation
        if monitoring_report['drift_analysis']['drift_detected']:
            return True
            
        if (monitoring_report.get('performance_analysis') and
            monitoring_report['performance_analysis']['performance_drop'] > self.update_threshold):
            return True
            
        return False
        
    @celery_app.task
    def update_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update the model with new data"""
        try:
            # Process new data
            X_processed, y_processed = self.data_processor.fit_transform(X, y)
            
            # Train new model
            training_results = self.model_trainer.train(
                X_processed,
                y_processed,
                **(model_params or {})
            )
            
            # Save new model
            self.model_trainer.save_model(str(self.model_dir / "model.joblib"))
            
            # Update metadata
            metadata = self._load_metadata()
            metadata['version'] += 1
            metadata['last_update'] = datetime.now().isoformat()
            metadata['performance_history'].append({
                'version': metadata['version'],
                'timestamp': datetime.now().isoformat(),
                'score': training_results['score']
            })
            self._save_metadata(metadata)
            
            # Generate and save report
            self.report_generator.save_json_report(
                {
                    'training_results': training_results,
                    'metadata': metadata
                },
                f"model_update_v{metadata['version']}"
            )
            
            return {
                'status': 'success',
                'version': metadata['version'],
                'training_results': training_results
            }
            
        except Exception as e:
            logging.error(f"Model update failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def get_update_status(self) -> Dict[str, Any]:
        """Get current model update status"""
        metadata = self._load_metadata()
        monitoring_path = self.model_dir / "monitoring.json"
        
        status = {
            'current_version': metadata['version'],
            'last_update': metadata['last_update'],
            'performance_history': metadata['performance_history']
        }
        
        if monitoring_path.exists():
            with open(monitoring_path, 'r') as f:
                status['monitoring'] = json.load(f)
                
        return status
