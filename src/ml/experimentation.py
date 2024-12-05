from typing import Dict, List, Optional, Union, Any
import json
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from src.ml.serving import ModelServer
from src.ml.monitoring import ModelMonitor
from src.models.experiment import Experiment, ExperimentVariant
from src.models.prediction import PredictionLog

class ABTesting:
    def __init__(
        self,
        db_session: Session,
        model_server: ModelServer,
        model_monitor: ModelMonitor
    ):
        self.db = db_session
        self.model_server = model_server
        self.model_monitor = model_monitor
        
    def create_experiment(
        self,
        name: str,
        description: str,
        variants: List[Dict[str, Any]],
        traffic_split: Optional[List[float]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Experiment:
        """Create a new A/B testing experiment"""
        # Validate traffic split
        if traffic_split is None:
            # Equal split between variants
            traffic_split = [1.0 / len(variants)] * len(variants)
        elif len(traffic_split) != len(variants):
            raise ValueError("Traffic split must match number of variants")
        elif sum(traffic_split) != 1.0:
            raise ValueError("Traffic split must sum to 1.0")
            
        # Create experiment
        experiment = Experiment(
            name=name,
            description=description,
            status='created',
            start_date=start_date or datetime.utcnow(),
            end_date=end_date,
            created_at=datetime.utcnow()
        )
        self.db.add(experiment)
        self.db.flush()
        
        # Create variants
        for variant, split in zip(variants, traffic_split):
            exp_variant = ExperimentVariant(
                experiment_id=experiment.id,
                name=variant['name'],
                model_id=variant['model_id'],
                model_version=variant.get('version'),
                traffic_percentage=split * 100,
                created_at=datetime.utcnow()
            )
            self.db.add(exp_variant)
            
        self.db.commit()
        return experiment
        
    def start_experiment(self, experiment_id: int) -> None:
        """Start an A/B testing experiment"""
        experiment = self.db.query(Experiment).filter_by(id=experiment_id).first()
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        experiment.status = 'running'
        experiment.started_at = datetime.utcnow()
        self.db.commit()
        
    def stop_experiment(self, experiment_id: int) -> None:
        """Stop an A/B testing experiment"""
        experiment = self.db.query(Experiment).filter_by(id=experiment_id).first()
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        experiment.status = 'completed'
        experiment.completed_at = datetime.utcnow()
        self.db.commit()
        
    def get_variant(self, experiment_id: int) -> ExperimentVariant:
        """Get a variant based on traffic split"""
        experiment = self.db.query(Experiment).filter_by(id=experiment_id).first()
        if not experiment or experiment.status != 'running':
            raise ValueError(f"No running experiment found with id {experiment_id}")
            
        variants = self.db.query(ExperimentVariant).filter_by(
            experiment_id=experiment_id
        ).all()
        
        # Select variant based on traffic split
        weights = [v.traffic_percentage / 100.0 for v in variants]
        selected_variant = random.choices(variants, weights=weights, k=1)[0]
        
        return selected_variant
        
    async def predict_with_experiment(
        self,
        experiment_id: int,
        data: Dict[str, Any],
        return_proba: bool = False
    ) -> Dict[str, Any]:
        """Make prediction using experiment variant"""
        # Get variant
        variant = self.get_variant(experiment_id)
        
        # Make prediction
        start_time = time.time()
        try:
            result = await self.model_server.predict(
                model_id=variant.model_id,
                data=data,
                version=variant.model_version,
                return_proba=return_proba
            )
            status = 'success'
        except Exception as e:
            result = {'error': str(e)}
            status = 'error'
            
        latency = time.time() - start_time
        
        # Log prediction with experiment info
        self.model_monitor.log_prediction(
            model_id=variant.model_id,
            version=variant.model_version,
            input_data=data,
            prediction=result,
            latency=latency,
            status=status,
            metadata={
                'experiment_id': experiment_id,
                'variant_id': variant.id
            }
        )
        
        return {
            'variant': variant.name,
            'result': result
        }
        
    def get_experiment_results(
        self,
        experiment_id: int,
        metric_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get experiment results and statistics"""
        experiment = self.db.query(Experiment).filter_by(id=experiment_id).first()
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        variants = self.db.query(ExperimentVariant).filter_by(
            experiment_id=experiment_id
        ).all()
        
        # Get predictions for each variant
        results = {}
        for variant in variants:
            # Query predictions
            query = self.db.query(PredictionLog).filter(
                PredictionLog.model_id == variant.model_id,
                PredictionLog.version == variant.model_version,
                PredictionLog.status == 'success'
            )
            
            if start_date:
                query = query.filter(PredictionLog.timestamp >= start_date)
            if end_date:
                query = query.filter(PredictionLog.timestamp <= end_date)
                
            predictions = query.all()
            
            # Calculate metrics
            metrics = {
                'total_predictions': len(predictions),
                'success_rate': len([p for p in predictions if p.status == 'success']) / len(predictions) if predictions else 0,
                'average_latency': np.mean([p.latency for p in predictions]) if predictions else 0
            }
            
            # Add custom metric if available
            if hasattr(variant, metric_name):
                metrics[metric_name] = getattr(variant, metric_name)
                
            results[variant.name] = metrics
            
        # Perform statistical analysis
        if len(variants) == 2:  # A/B test
            control, treatment = variants
            control_data = [p.latency for p in results[control.name]['predictions']]
            treatment_data = [p.latency for p in results[treatment.name]['predictions']]
            
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(control_data, treatment_data)
            
            results['statistical_analysis'] = {
                'test_type': 't-test',
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
        return {
            'experiment': {
                'id': experiment.id,
                'name': experiment.name,
                'status': experiment.status,
                'duration': (experiment.completed_at or datetime.utcnow()) - experiment.start_date
            },
            'variants': results
        }
        
    def get_winning_variant(
        self,
        experiment_id: int,
        metric_name: str,
        higher_is_better: bool = True
    ) -> Optional[ExperimentVariant]:
        """Determine the winning variant based on metric"""
        results = self.get_experiment_results(experiment_id, metric_name)
        
        if not results['variants']:
            return None
            
        # Find variant with best metric
        variants = results['variants']
        best_variant = max(
            variants.items(),
            key=lambda x: x[1][metric_name] if higher_is_better else -x[1][metric_name]
        )
        
        return self.db.query(ExperimentVariant).filter_by(
            experiment_id=experiment_id,
            name=best_variant[0]
        ).first()
        
    def promote_variant(
        self,
        experiment_id: int,
        variant_id: int
    ) -> None:
        """Promote a variant to production"""
        variant = self.db.query(ExperimentVariant).filter_by(
            experiment_id=experiment_id,
            id=variant_id
        ).first()
        
        if not variant:
            raise ValueError(f"Variant {variant_id} not found in experiment {experiment_id}")
            
        # Update model version stage
        self.model_server.registry.update_model_stage(
            model_id=variant.model_id,
            version=variant.model_version,
            stage='production'
        )
        
        # Stop experiment
        self.stop_experiment(experiment_id)
        
    def get_active_experiments(self) -> List[Dict[str, Any]]:
        """Get all active experiments"""
        experiments = self.db.query(Experiment).filter_by(status='running').all()
        
        return [{
            'id': exp.id,
            'name': exp.name,
            'description': exp.description,
            'start_date': exp.start_date.isoformat(),
            'duration': (datetime.utcnow() - exp.start_date).days,
            'variants': [
                {
                    'name': v.name,
                    'traffic_percentage': v.traffic_percentage
                }
                for v in exp.variants
            ]
        } for exp in experiments]
