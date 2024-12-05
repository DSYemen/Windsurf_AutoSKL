from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import os
import json
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from src.ml.pipeline import AutoMLPipeline
from src.ml.validation import ModelValidator

logger = logging.getLogger(__name__)

class DistributedTrainer:
    def __init__(
        self,
        task_type: str,
        target_column: str,
        num_workers: int = -1,
        use_ray: bool = True,
        optimization_metric: str = None,
        max_trials: int = 100,
        time_budget_seconds: Optional[int] = None
    ):
        self.task_type = task_type
        self.target_column = target_column
        self.num_workers = num_workers if num_workers > 0 else os.cpu_count()
        self.use_ray = use_ray
        self.optimization_metric = optimization_metric
        self.max_trials = max_trials
        self.time_budget_seconds = time_budget_seconds
        self.best_model = None
        self.best_pipeline = None
        self.training_results = None
        
        # Initialize Ray if needed
        if self.use_ray and not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            
    def _get_default_metric(self) -> str:
        """Get default optimization metric based on task type"""
        metrics = {
            'classification': 'accuracy',
            'regression': 'neg_mean_squared_error',
            'clustering': 'silhouette_score',
            'dimensionality_reduction': 'explained_variance'
        }
        return metrics.get(self.task_type, 'accuracy')
        
    def _train_model_ray(
        self,
        config: Dict[str, Any],
        data: Dict[str, Any]
    ) -> None:
        """Training function for Ray Tune"""
        # Extract data
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
        
        # Create and train model
        model = config['model_class'](**config['model_params'])
        model.fit(X_train, y_train)
        
        # Evaluate
        if self.task_type == 'classification':
            score = model.score(X_val, y_val)
        else:
            y_pred = model.predict(X_val)
            score = -np.mean((y_val - y_pred) ** 2)  # Negative MSE
            
        # Report to Ray Tune
        tune.report(score=score)
        
    def _parallel_cv_train(
        self,
        model_config: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        cv_indices: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Dict[str, float]]:
        """Train and evaluate model in parallel using CV folds"""
        def _train_fold(fold_data):
            train_idx, val_idx = fold_data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create and train model
            model = model_config['model_class'](**model_config['model_params'])
            model.fit(X_train, y_train)
            
            # Evaluate
            validator = ModelValidator(self.task_type)
            metrics = validator._calculate_metrics(y_val, model.predict(X_val))
            
            return metrics
            
        # Use ProcessPoolExecutor for parallel training
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(_train_fold, cv_indices))
            
        return results
        
    def train_distributed(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_configs: List[Dict[str, Any]],
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """Train models in a distributed manner"""
        validator = ModelValidator(
            task_type=self.task_type,
            cv_folds=cv_folds
        )
        cv_splitter = validator._get_cv_splitter()
        
        if self.use_ray:
            # Ray Tune setup
            metric = self.optimization_metric or self._get_default_metric()
            scheduler = ASHAScheduler(
                max_t=cv_folds,
                grace_period=1,
                reduction_factor=2
            )
            
            search_space = {
                model_config['name']: {
                    'model_class': tune.choice([model_config['class']]),
                    'model_params': model_config['params']
                }
                for model_config in model_configs
            }
            
            # Prepare data for Ray
            data = {
                'X_train': X,
                'y_train': y,
                'X_val': X,
                'y_val': y
            }
            
            # Run hyperparameter optimization
            analysis = tune.run(
                lambda config: self._train_model_ray(config, data),
                config=search_space,
                metric='score',
                mode='max',
                num_samples=self.max_trials,
                scheduler=scheduler,
                search_alg=OptunaSearch(),
                time_budget_s=self.time_budget_seconds,
                verbose=1
            )
            
            # Get best trial
            best_trial = analysis.best_trial
            best_config = best_trial.config
            best_score = best_trial.last_result['score']
            
            # Create final model with best config
            self.best_model = best_config['model_class'](**best_config['model_params'])
            self.best_model.fit(X, y)
            
        else:
            # Parallel training with ProcessPoolExecutor
            best_score = float('-inf')
            best_config = None
            
            for model_config in model_configs:
                # Get CV indices
                cv_indices = list(cv_splitter.split(X, y))
                
                # Train in parallel
                fold_results = self._parallel_cv_train(
                    model_config,
                    X,
                    y,
                    cv_indices
                )
                
                # Calculate average score
                avg_score = np.mean([
                    result[self.optimization_metric or self._get_default_metric()]
                    for result in fold_results
                ])
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_config = model_config
                    
            # Train final model with best config
            self.best_model = best_config['model_class'](**best_config['model_params'])
            self.best_model.fit(X, y)
            
        # Validate final model
        validation_results = validator.validate_model(
            self.best_model,
            X,
            y,
            feature_names=X.columns.tolist()
        )
        
        self.training_results = {
            'best_model_config': best_config,
            'best_score': best_score,
            'validation_results': validation_results
        }
        
        return self.training_results
        
    def predict_distributed(
        self,
        X: pd.DataFrame,
        batch_size: int = 1000
    ) -> np.ndarray:
        """Make predictions in parallel"""
        if self.best_model is None:
            raise ValueError("No model available. Train the model first.")
            
        def _predict_batch(batch_data):
            return self.best_model.predict(batch_data)
            
        # Split data into batches
        n_samples = len(X)
        n_batches = (n_samples + batch_size - 1) // batch_size
        batches = [
            X.iloc[i * batch_size:(i + 1) * batch_size]
            for i in range(n_batches)
        ]
        
        # Make predictions in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            predictions = list(executor.map(_predict_batch, batches))
            
        return np.concatenate(predictions)
        
    def save_model(self, path: str) -> None:
        """Save the trained model and metadata"""
        if self.best_model is None:
            raise ValueError("No model available to save.")
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model using joblib
        from joblib import dump
        dump(self.best_model, f"{path}_model.joblib")
        
        # Save metadata
        metadata = {
            'task_type': self.task_type,
            'target_column': self.target_column,
            'optimization_metric': self.optimization_metric,
            'training_results': {
                k: v for k, v in self.training_results.items()
                if isinstance(v, (str, int, float, bool, list, dict))
            }
        }
        
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
    @classmethod
    def load_model(cls, path: str) -> 'DistributedTrainer':
        """Load a saved model and metadata"""
        # Load metadata
        with open(f"{path}_metadata.json", 'r') as f:
            metadata = json.load(f)
            
        # Create instance
        instance = cls(
            task_type=metadata['task_type'],
            target_column=metadata['target_column'],
            optimization_metric=metadata['optimization_metric']
        )
        
        # Load model
        from joblib import load
        instance.best_model = load(f"{path}_model.joblib")
        instance.training_results = metadata['training_results']
        
        return instance
