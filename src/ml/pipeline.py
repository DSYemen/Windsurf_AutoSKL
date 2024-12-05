from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder, BinaryEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from src.ml.preprocessing import DataPreprocessor
from src.ml.optimization import ModelOptimizer

class AutoMLPipeline:
    def __init__(
        self,
        task_type: str,
        target_column: str,
        optimization_metric: str = None,
        max_trials: int = 100,
        feature_selection: bool = True,
        feature_engineering: bool = True
    ):
        self.task_type = task_type
        self.target_column = target_column
        self.optimization_metric = optimization_metric or self._default_metric()
        self.max_trials = max_trials
        self.feature_selection = feature_selection
        self.feature_engineering = feature_engineering
        self.preprocessor = DataPreprocessor()
        self.best_pipeline = None
        self.best_model = None
        self.feature_importance = None
        
    def _default_metric(self) -> str:
        """Get default optimization metric based on task type"""
        metrics = {
            'classification': 'accuracy',
            'regression': 'neg_mean_squared_error',
            'clustering': 'silhouette_score',
            'dimensionality_reduction': 'explained_variance'
        }
        return metrics.get(self.task_type, 'accuracy')
        
    def _create_feature_engineering_pipeline(
        self,
        numeric_features: List[str],
        categorical_features: List[str]
    ) -> ColumnTransformer:
        """Create feature engineering pipeline"""
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', TargetEncoder())
        ])
        
        # Add polynomial features for numeric columns if enabled
        if self.feature_engineering:
            from sklearn.preprocessing import PolynomialFeatures
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('scaler', StandardScaler())
            ])
        
        return ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
    def _get_algorithms(self) -> List[Dict[str, Any]]:
        """Get list of algorithms to try based on task type"""
        if self.task_type == 'classification':
            return [
                {
                    'name': 'RandomForestClassifier',
                    'class': 'sklearn.ensemble.RandomForestClassifier',
                    'params': {
                        'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                        'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                        'min_samples_split': {'type': 'int', 'low': 2, 'high': 20}
                    }
                },
                {
                    'name': 'XGBClassifier',
                    'class': 'xgboost.XGBClassifier',
                    'params': {
                        'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                        'max_depth': {'type': 'int', 'low': 3, 'high': 12},
                        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3}
                    }
                },
                {
                    'name': 'LGBMClassifier',
                    'class': 'lightgbm.LGBMClassifier',
                    'params': {
                        'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                        'max_depth': {'type': 'int', 'low': 3, 'high': 12},
                        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3}
                    }
                }
            ]
        elif self.task_type == 'regression':
            return [
                {
                    'name': 'RandomForestRegressor',
                    'class': 'sklearn.ensemble.RandomForestRegressor',
                    'params': {
                        'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                        'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                        'min_samples_split': {'type': 'int', 'low': 2, 'high': 20}
                    }
                },
                {
                    'name': 'XGBRegressor',
                    'class': 'xgboost.XGBRegressor',
                    'params': {
                        'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                        'max_depth': {'type': 'int', 'low': 3, 'high': 12},
                        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3}
                    }
                },
                {
                    'name': 'LGBMRegressor',
                    'class': 'lightgbm.LGBMRegressor',
                    'params': {
                        'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                        'max_depth': {'type': 'int', 'low': 3, 'high': 12},
                        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3}
                    }
                }
            ]
        # Add more algorithms for other task types
        return []
        
    def fit(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fit automated ML pipeline"""
        # Split features and target
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]
        
        # Analyze features
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create feature engineering pipeline
        feature_pipeline = self._create_feature_engineering_pipeline(
            numeric_features,
            categorical_features
        )
        
        # Feature selection if enabled
        if self.feature_selection:
            if self.task_type == 'classification':
                selector = SelectKBest(score_func=mutual_info_classif, k='all')
            else:
                selector = SelectKBest(score_func=mutual_info_regression, k='all')
                
            feature_pipeline = Pipeline([
                ('features', feature_pipeline),
                ('selection', selector)
            ])
        
        # Try different algorithms
        best_score = float('-inf')
        best_algorithm = None
        best_params = None
        
        for algorithm in self._get_algorithms():
            # Import algorithm class dynamically
            from importlib import import_module
            module_path, class_name = algorithm['class'].rsplit('.', 1)
            module = import_module(module_path)
            model_class = getattr(module, class_name)
            
            # Create and optimize model
            optimizer = ModelOptimizer(
                model_class=model_class,
                param_space=algorithm['params'],
                metric=self.optimization_metric,
                n_trials=self.max_trials
            )
            
            # Create full pipeline
            pipeline = Pipeline([
                ('preprocessing', feature_pipeline),
                ('model', model_class())
            ])
            
            # Optimize
            result = optimizer.optimize(X, y)
            
            if result['best_value'] > best_score:
                best_score = result['best_value']
                best_algorithm = algorithm
                best_params = result['best_params']
                self.best_pipeline = pipeline
                
        # Fit best pipeline with best parameters
        self.best_model = best_algorithm['class'](**best_params)
        self.best_pipeline.set_params(**{
            f'model__{k}': v for k, v in best_params.items()
        })
        self.best_pipeline.fit(X, y)
        
        # Calculate feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                X.columns,
                self.best_model.feature_importances_
            ))
        
        return {
            'best_algorithm': best_algorithm['name'],
            'best_parameters': best_params,
            'best_score': best_score,
            'feature_importance': self.feature_importance
        }
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using best pipeline"""
        if self.best_pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        return self.best_pipeline.predict(data)
        
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available"""
        return self.feature_importance
