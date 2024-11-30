from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    make_scorer, accuracy_score, mean_squared_error, r2_score,
    f1_score, precision_score, recall_score, roc_auc_score
)
import optuna
from sklearn.base import BaseEstimator
import sklearn.ensemble as ensemble
import sklearn.linear_model as linear_model
import sklearn.tree as tree
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.neighbors as neighbors
import sklearn.neural_network as neural_network
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import joblib
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

class ModelTrainer:
    def __init__(self):
        self.task_type = None
        self.best_model = None
        self.best_score = None
        self.best_params = None
        self.label_encoder = LabelEncoder()
        
    def _detect_task_type(self, y: np.ndarray) -> str:
        """Detect if the task is classification or regression"""
        unique_values = np.unique(y)
        if len(unique_values) < 10 or isinstance(y[0], (str, bool)):
            return 'classification'
        if np.all(np.mod(y, 1) == 0) and len(unique_values) < 100:
            return 'classification'
        return 'regression'
        
    def _get_models(self, task_type: str, data_size: int) -> Dict[str, Any]:
        """Get dictionary of models based on task type and data characteristics"""
        models = {}
        
        # Tree-based Models (good for most tasks)
        models['random_forest'] = (
            ensemble.RandomForestClassifier if task_type == 'classification'
            else ensemble.RandomForestRegressor
        )
        models['gradient_boosting'] = (
            ensemble.GradientBoostingClassifier if task_type == 'classification'
            else ensemble.GradientBoostingRegressor
        )
        models['xgboost'] = XGBClassifier if task_type == 'classification' else XGBRegressor
        models['lightgbm'] = LGBMClassifier if task_type == 'classification' else LGBMRegressor
        
        # Linear Models (good for high-dimensional data)
        if task_type == 'classification':
            models['logistic_regression'] = linear_model.LogisticRegression
            models['sgd_classifier'] = linear_model.SGDClassifier
        else:
            models['linear_regression'] = linear_model.LinearRegression
            models['lasso'] = linear_model.Lasso
            models['ridge'] = linear_model.Ridge
            models['elastic_net'] = linear_model.ElasticNet
        
        # Support Vector Machines (good for non-linear relationships)
        if data_size < 10000:  # SVMs don't scale well to large datasets
            if task_type == 'classification':
                models['svc'] = svm.SVC
            else:
                models['svr'] = svm.SVR
        
        # K-Nearest Neighbors (good for non-linear relationships)
        if data_size < 100000:
            models['knn'] = (
                neighbors.KNeighborsClassifier if task_type == 'classification'
                else neighbors.KNeighborsRegressor
            )
        
        # Neural Networks (good for complex patterns)
        if data_size >= 1000:
            models['neural_network'] = (
                neural_network.MLPClassifier if task_type == 'classification'
                else neural_network.MLPRegressor
            )
        
        # Decision Trees (good for interpretability)
        models['decision_tree'] = (
            tree.DecisionTreeClassifier if task_type == 'classification'
            else tree.DecisionTreeRegressor
        )
        
        # Naive Bayes (good for text classification)
        if task_type == 'classification':
            models['gaussian_nb'] = naive_bayes.GaussianNB
        
        return models
            
    def _get_metric(self, task_type: str):
        """Get appropriate metrics based on task type"""
        if task_type == 'classification':
            return {
                'accuracy': make_scorer(accuracy_score),
                'f1': make_scorer(f1_score, average='weighted'),
                'precision': make_scorer(precision_score, average='weighted'),
                'recall': make_scorer(recall_score, average='weighted')
            }
        return {
            'r2': make_scorer(r2_score),
            'mse': make_scorer(mean_squared_error, greater_is_better=False)
        }
        
    def _get_hyperparameter_space(self, model_class: Any) -> Dict[str, Any]:
        """Define hyperparameter search space based on model type"""
        params = {}
        
        # Tree-based Models
        if model_class in [ensemble.RandomForestClassifier, ensemble.RandomForestRegressor]:
            params.update({
                'n_estimators': ('int', 50, 300),
                'max_depth': ('int', 3, 20),
                'min_samples_split': ('int', 2, 20),
                'min_samples_leaf': ('int', 1, 10)
            })
        elif model_class in [ensemble.GradientBoostingClassifier, ensemble.GradientBoostingRegressor]:
            params.update({
                'n_estimators': ('int', 50, 300),
                'max_depth': ('int', 3, 10),
                'learning_rate': ('float', 0.01, 0.3),
                'min_samples_split': ('int', 2, 20)
            })
        elif model_class in [XGBClassifier, XGBRegressor, LGBMClassifier, LGBMRegressor]:
            params.update({
                'n_estimators': ('int', 50, 300),
                'max_depth': ('int', 3, 10),
                'learning_rate': ('float', 0.01, 0.3)
            })
            
        # Linear Models
        elif model_class in [linear_model.LogisticRegression, linear_model.LinearRegression]:
            params.update({
                'fit_intercept': ('categorical', [True, False])
            })
        elif model_class in [linear_model.Lasso, linear_model.Ridge, linear_model.ElasticNet]:
            params.update({
                'alpha': ('float', 0.0001, 10.0),
                'fit_intercept': ('categorical', [True, False])
            })
            
        # SVM
        elif model_class in [svm.SVC, svm.SVR]:
            params.update({
                'C': ('float', 0.1, 10.0),
                'kernel': ('categorical', ['rbf', 'linear', 'poly']),
                'gamma': ('categorical', ['scale', 'auto'])
            })
            
        # KNN
        elif model_class in [neighbors.KNeighborsClassifier, neighbors.KNeighborsRegressor]:
            params.update({
                'n_neighbors': ('int', 3, 20),
                'weights': ('categorical', ['uniform', 'distance'])
            })
            
        # Neural Networks
        elif model_class in [neural_network.MLPClassifier, neural_network.MLPRegressor]:
            params.update({
                'hidden_layer_sizes': ('categorical', [(50,), (100,), (50, 50), (100, 50)]),
                'learning_rate_init': ('float', 0.0001, 0.1),
                'max_iter': ('int', 100, 500)
            })
            
        # Decision Trees
        elif model_class in [tree.DecisionTreeClassifier, tree.DecisionTreeRegressor]:
            params.update({
                'max_depth': ('int', 3, 20),
                'min_samples_split': ('int', 2, 20),
                'min_samples_leaf': ('int', 1, 10)
            })
            
        return params
        
    def _optimize_hyperparameters(
        self, 
        model_class: Any,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 100
    ) -> Tuple[Dict[str, Any], float]:
        """Optimize hyperparameters using Optuna"""
        param_space = self._get_hyperparameter_space(model_class)
        metrics = self._get_metric(self.task_type)
        primary_metric = 'accuracy' if self.task_type == 'classification' else 'r2'
        
        def objective(trial):
            params = {}
            
            # Generate parameters based on defined space
            for param_name, param_config in param_space.items():
                param_type, *param_args = param_config
                if param_type == 'int':
                    params[param_name] = trial.suggest_int(param_name, param_args[0], param_args[1])
                elif param_type == 'float':
                    params[param_name] = trial.suggest_float(param_name, param_args[0], param_args[1])
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, param_args[0])
            
            model = model_class(**params)
            scores = {
                metric_name: cross_val_score(
                    model, X, y,
                    scoring=metric_scorer,
                    cv=5
                ).mean()
                for metric_name, metric_scorer in metrics.items()
            }
            
            # Store all metrics in trial user attributes
            for metric_name, score in scores.items():
                trial.set_user_attr(metric_name, score)
            
            return scores[primary_metric]
            
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params, study.best_value
        
    def _evaluate_model_suitability(
        self,
        model_class: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> float:
        """Evaluate how suitable a model is for the given data"""
        n_samples, n_features = X.shape
        
        # Base score starts at 1.0
        score = 1.0
        
        # Adjust score based on data characteristics
        if model_class in [svm.SVC, svm.SVR]:
            # SVMs don't work well with large datasets
            score *= min(1.0, 10000 / n_samples)
            
        elif model_class in [neighbors.KNeighborsClassifier, neighbors.KNeighborsRegressor]:
            # KNN works better with smaller feature spaces
            score *= min(1.0, 100 / n_features)
            
        elif model_class in [linear_model.LogisticRegression, linear_model.LinearRegression]:
            # Linear models work better with more samples than features
            score *= min(1.0, n_samples / (n_features * 10))
            
        elif model_class in [neural_network.MLPClassifier, neural_network.MLPRegressor]:
            # Neural networks need more data
            score *= min(1.0, n_samples / 1000)
            
        return score
        
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_types: Optional[List[str]] = None,
        n_trials: int = 100
    ) -> Dict[str, Any]:
        """Train and optimize multiple models, select the best one"""
        self.task_type = self._detect_task_type(y)
        
        # Encode labels for classification
        if self.task_type == 'classification':
            y = self.label_encoder.fit_transform(y)
        
        # Get available models
        models = self._get_models(self.task_type, len(X))
        if model_types:
            models = {k: v for k, v in models.items() if k in model_types}
            
        # Calculate model suitability scores
        suitability_scores = {
            model_name: self._evaluate_model_suitability(model_class, X, y)
            for model_name, model_class in models.items()
        }
        
        # Sort models by suitability
        sorted_models = sorted(
            suitability_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Try models in order of suitability
        results = []
        for model_name, suitability in sorted_models[:3]:  # Try top 3 most suitable models
            model_class = models[model_name]
            logging.info(f"Training {model_name} (suitability: {suitability:.2f})")
            
            try:
                params, score = self._optimize_hyperparameters(
                    model_class, X, y, n_trials
                )
                
                model = model_class(**params)
                model.fit(X, y)
                
                results.append({
                    'model_type': model_name,
                    'model': model,
                    'params': params,
                    'score': score,
                    'suitability': suitability
                })
                
            except Exception as e:
                logging.error(f"Error training {model_name}: {str(e)}")
                continue
        
        if not results:
            raise ValueError("No models could be trained successfully")
            
        # Select best model
        best_result = max(results, key=lambda x: x['score'])
        self.best_model = best_result['model']
        self.best_score = best_result['score']
        self.best_params = best_result['params']
        
        return {
            'model_type': best_result['model_type'],
            'score': self.best_score,
            'parameters': self.best_params,
            'all_results': results
        }
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the best model"""
        if self.best_model is None:
            raise ValueError("Model has not been trained yet")
            
        predictions = self.best_model.predict(X)
        
        # Decode predictions for classification
        if self.task_type == 'classification':
            predictions = self.label_encoder.inverse_transform(predictions)
            
        return predictions
        
    def save_model(self, path: str):
        """Save the trained model to disk"""
        if self.best_model is None:
            raise ValueError("No model to save")
            
        joblib.dump({
            'model': self.best_model,
            'task_type': self.task_type,
            'label_encoder': self.label_encoder
        }, path)
        
    @classmethod
    def load_model(cls, path: str) -> 'ModelTrainer':
        """Load a trained model from disk"""
        data = joblib.load(path)
        trainer = cls()
        trainer.best_model = data['model']
        trainer.task_type = data['task_type']
        trainer.label_encoder = data['label_encoder']
        return trainer
