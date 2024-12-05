from typing import Dict, List, Optional, Any, Callable
import optuna
from optuna.trial import Trial
import numpy as np
from sklearn.model_selection import cross_val_score
import logging

class ModelOptimizer:
    def __init__(
        self,
        model_class: Any,
        param_space: Dict[str, Any],
        metric: str = 'accuracy',
        direction: str = 'maximize',
        n_trials: int = 100,
        cv: int = 5,
        random_state: int = 42
    ):
        self.model_class = model_class
        self.param_space = param_space
        self.metric = metric
        self.direction = direction
        self.n_trials = n_trials
        self.cv = cv
        self.random_state = random_state
        self.study = None
        self.best_params = None
        self.optimization_history = []
        
    def _create_objective(self, X, y):
        def objective(trial: Trial) -> float:
            params = {}
            
            # Generate parameters based on param_space definition
            for param_name, param_config in self.param_space.items():
                if param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['values']
                    )
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        step=param_config.get('step', 1),
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
            
            # Create and evaluate model with suggested parameters
            model = self.model_class(**params)
            scores = cross_val_score(
                model, X, y,
                scoring=self.metric,
                cv=self.cv,
                n_jobs=-1
            )
            
            return scores.mean()
            
        return objective
    
    def optimize(self, X, y) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        study = optuna.create_study(direction=self.direction)
        objective = self._create_objective(X, y)
        
        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        self.study = study
        self.best_params = study.best_params
        
        # Store optimization history
        self.optimization_history = [
            {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state
            }
            for trial in study.trials
        ]
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }
    
    def get_param_importances(self) -> Dict[str, float]:
        """Get parameter importances"""
        if self.study is None:
            raise ValueError("Must run optimize() first")
        
        importances = optuna.importance.get_param_importances(self.study)
        return dict(importances)
    
    @staticmethod
    def get_default_param_space(algorithm: str) -> Dict[str, Dict]:
        """Get default parameter space for common algorithms"""
        spaces = {
            'RandomForestClassifier': {
                'n_estimators': {
                    'type': 'int',
                    'low': 50,
                    'high': 300,
                    'step': 10
                },
                'max_depth': {
                    'type': 'int',
                    'low': 3,
                    'high': 20
                },
                'min_samples_split': {
                    'type': 'int',
                    'low': 2,
                    'high': 20
                },
                'min_samples_leaf': {
                    'type': 'int',
                    'low': 1,
                    'high': 10
                }
            },
            'XGBClassifier': {
                'n_estimators': {
                    'type': 'int',
                    'low': 50,
                    'high': 300,
                    'step': 10
                },
                'max_depth': {
                    'type': 'int',
                    'low': 3,
                    'high': 12
                },
                'learning_rate': {
                    'type': 'float',
                    'low': 0.01,
                    'high': 0.3,
                    'log': True
                },
                'subsample': {
                    'type': 'float',
                    'low': 0.5,
                    'high': 1.0
                },
                'colsample_bytree': {
                    'type': 'float',
                    'low': 0.5,
                    'high': 1.0
                }
            },
            'LGBMClassifier': {
                'n_estimators': {
                    'type': 'int',
                    'low': 50,
                    'high': 300,
                    'step': 10
                },
                'max_depth': {
                    'type': 'int',
                    'low': 3,
                    'high': 12
                },
                'learning_rate': {
                    'type': 'float',
                    'low': 0.01,
                    'high': 0.3,
                    'log': True
                },
                'num_leaves': {
                    'type': 'int',
                    'low': 20,
                    'high': 100
                },
                'feature_fraction': {
                    'type': 'float',
                    'low': 0.5,
                    'high': 1.0
                }
            }
        }
        
        return spaces.get(algorithm, {})
