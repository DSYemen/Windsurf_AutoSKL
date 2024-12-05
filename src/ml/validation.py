from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    cross_val_score, cross_validate,
    StratifiedKFold, KFold,
    train_test_split
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error,
    roc_auc_score, confusion_matrix
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ModelValidator:
    def __init__(
        self,
        task_type: str,
        cv_folds: int = 5,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        self.task_type = task_type
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state
        self.validation_results = None
        
    def _get_cv_splitter(self):
        """Get appropriate cross-validation splitter"""
        if self.task_type == 'classification':
            return StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state
            )
        return KFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
    def _get_scoring_metrics(self) -> Dict[str, str]:
        """Get appropriate scoring metrics based on task type"""
        if self.task_type == 'classification':
            return {
                'accuracy': 'accuracy',
                'precision_weighted': 'precision_weighted',
                'recall_weighted': 'recall_weighted',
                'f1_weighted': 'f1_weighted',
                'roc_auc_ovr': 'roc_auc_ovr'
            }
        elif self.task_type == 'regression':
            return {
                'r2': 'r2',
                'neg_mean_squared_error': 'neg_mean_squared_error',
                'neg_mean_absolute_error': 'neg_mean_absolute_error',
                'neg_root_mean_squared_error': 'neg_root_mean_squared_error'
            }
        return {}
        
    def validate_model(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive model validation"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if self.task_type == 'classification' else None
        )
        
        # Perform cross-validation
        cv_splitter = self._get_cv_splitter()
        scoring = self._get_scoring_metrics()
        
        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv_splitter,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        # Train final model and get predictions
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate test metrics
        test_metrics = self._calculate_metrics(y_test, y_pred)
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
            feature_importance = dict(zip(feature_names, model.feature_importances_))
        
        # Store validation results
        self.validation_results = {
            'cv_results': {
                metric: {
                    'train_mean': cv_results[f'train_{metric}'].mean(),
                    'train_std': cv_results[f'train_{metric}'].std(),
                    'test_mean': cv_results[f'test_{metric}'].mean(),
                    'test_std': cv_results[f'test_{metric}'].std()
                }
                for metric in scoring.keys()
            },
            'test_metrics': test_metrics,
            'feature_importance': feature_importance
        }
        
        return self.validation_results
        
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate appropriate metrics based on task type"""
        if self.task_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted')
            }
            
            # Add ROC-AUC if binary classification
            if len(np.unique(y_true)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
                
            return metrics
            
        elif self.task_type == 'regression':
            return {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
            
        return {}
        
    def plot_validation_results(self) -> Dict[str, go.Figure]:
        """Create visualization plots for validation results"""
        if not self.validation_results:
            raise ValueError("No validation results available. Run validate_model first.")
            
        plots = {}
        
        # CV Results comparison
        cv_metrics = list(self.validation_results['cv_results'].keys())
        n_metrics = len(cv_metrics)
        
        fig = make_subplots(
            rows=n_metrics,
            cols=1,
            subplot_titles=cv_metrics
        )
        
        for i, metric in enumerate(cv_metrics, 1):
            results = self.validation_results['cv_results'][metric]
            
            fig.add_trace(
                go.Bar(
                    x=['Train', 'Test'],
                    y=[results['train_mean'], results['test_mean']],
                    error_y=dict(
                        type='data',
                        array=[results['train_std'], results['test_std']]
                    ),
                    name=metric
                ),
                row=i,
                col=1
            )
            
        fig.update_layout(height=300 * n_metrics, showlegend=False)
        plots['cv_comparison'] = fig
        
        # Feature importance plot
        if self.validation_results['feature_importance']:
            importance_df = pd.DataFrame({
                'Feature': list(self.validation_results['feature_importance'].keys()),
                'Importance': list(self.validation_results['feature_importance'].values())
            }).sort_values('Importance', ascending=True)
            
            fig = go.Figure(go.Bar(
                x=importance_df['Importance'],
                y=importance_df['Feature'],
                orientation='h'
            ))
            
            fig.update_layout(
                title='Feature Importance',
                xaxis_title='Importance',
                yaxis_title='Feature',
                height=max(400, len(importance_df) * 20)
            )
            
            plots['feature_importance'] = fig
            
        return plots
        
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results"""
        if not self.validation_results:
            raise ValueError("No validation results available. Run validate_model first.")
            
        cv_summary = {
            metric: {
                'train_score': f"{results['train_mean']:.4f} ± {results['train_std']:.4f}",
                'test_score': f"{results['test_mean']:.4f} ± {results['test_std']:.4f}",
                'difference': results['test_mean'] - results['train_mean']
            }
            for metric, results in self.validation_results['cv_results'].items()
        }
        
        return {
            'cross_validation': cv_summary,
            'test_metrics': self.validation_results['test_metrics'],
            'n_features': len(self.validation_results['feature_importance'])
            if self.validation_results['feature_importance']
            else None
        }
