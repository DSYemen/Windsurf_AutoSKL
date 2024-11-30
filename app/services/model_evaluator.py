from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error,
    r2_score, mean_absolute_error, explained_variance_score
)
from sklearn.model_selection import learning_curve
import shap
import lime
import lime.lime_tabular
from yellowbrick.classifier import ROCAUC, PrecisionRecallCurve
from yellowbrick.regressor import ResidualsPlot
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import logging

class ModelEvaluator:
    def __init__(self):
        self.task_type = None
        self.feature_names = None
        self.class_names = None
        self.shap_explainer = None
        self.lime_explainer = None
        
    def setup(
        self,
        task_type: str,
        feature_names: List[str],
        class_names: Optional[List[str]] = None
    ):
        """Setup the evaluator with model information"""
        self.task_type = task_type
        self.feature_names = feature_names
        self.class_names = class_names
        
    def evaluate_model(
        self,
        model: Any,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        evaluation = {}
        
        # Performance metrics
        evaluation['metrics'] = self._calculate_metrics(
            model, X_test, y_test
        )
        
        # Learning curves
        evaluation['learning_curves'] = self._generate_learning_curves(
            model, X_train, y_train
        )
        
        # Feature importance
        evaluation['feature_importance'] = self._analyze_feature_importance(
            model, X_train
        )
        
        # Model explanations
        evaluation['explanations'] = self._setup_explainers(
            model, X_train
        )
        
        # Performance visualizations
        evaluation['visualizations'] = self._generate_visualizations(
            model, X_test, y_test
        )
        
        return evaluation
        
    def _calculate_metrics(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """Calculate performance metrics based on task type"""
        metrics = {}
        y_pred = model.predict(X_test)
        
        if self.task_type == 'classification':
            metrics.update({
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            })
            
            # ROC AUC for binary classification
            if len(np.unique(y_test)) == 2:
                try:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
                except:
                    pass
                    
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
        else:  # regression
            metrics.update({
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'explained_variance': explained_variance_score(y_test, y_pred)
            })
            
        return metrics
        
    def _generate_learning_curves(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, List[float]]:
        """Generate learning curves to analyze model learning"""
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                model, X, y,
                train_sizes=np.linspace(0.1, 1.0, 10),
                cv=5,
                n_jobs=-1
            )
            
            return {
                'train_sizes': train_sizes.tolist(),
                'train_scores_mean': train_scores.mean(axis=1).tolist(),
                'train_scores_std': train_scores.std(axis=1).tolist(),
                'val_scores_mean': val_scores.mean(axis=1).tolist(),
                'val_scores_std': val_scores.std(axis=1).tolist()
            }
        except Exception as e:
            logging.error(f"Error generating learning curves: {str(e)}")
            return {}
            
    def _analyze_feature_importance(
        self,
        model: Any,
        X: np.ndarray
    ) -> Dict[str, float]:
        """Analyze feature importance using various methods"""
        importance_dict = {}
        
        try:
            # Try getting feature importance directly from model
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_).mean(axis=0) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
            else:
                return importance_dict
                
            # Create feature importance dictionary
            for idx, importance_value in enumerate(importance):
                feature_name = self.feature_names[idx] if self.feature_names else f"feature_{idx}"
                importance_dict[feature_name] = float(importance_value)
                
            # Normalize importance values
            total = sum(importance_dict.values())
            importance_dict = {k: v/total for k, v in importance_dict.items()}
            
        except Exception as e:
            logging.error(f"Error analyzing feature importance: {str(e)}")
            
        return importance_dict
        
    def _setup_explainers(
        self,
        model: Any,
        X: np.ndarray
    ) -> Dict[str, Any]:
        """Setup SHAP and LIME explainers"""
        try:
            # Setup SHAP explainer
            self.shap_explainer = shap.TreeExplainer(model) if hasattr(model, 'predict_proba') else shap.KernelExplainer(model.predict, X)
            
            # Setup LIME explainer
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X,
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode='classification' if self.task_type == 'classification' else 'regression'
            )
            
            return {'status': 'success'}
        except Exception as e:
            logging.error(f"Error setting up explainers: {str(e)}")
            return {'status': 'error', 'message': str(e)}
            
    def _generate_visualizations(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, str]:
        """Generate performance visualizations"""
        visualizations = {}
        
        try:
            plt.figure(figsize=(10, 6))
            
            if self.task_type == 'classification':
                if len(np.unique(y)) == 2:
                    # ROC curve
                    viz = ROCAUC(model)
                    viz.fit(X, y)
                    viz.score(X, y)
                    visualizations['roc_curve'] = self._fig_to_base64(viz.fig)
                    
                    # Precision-Recall curve
                    viz = PrecisionRecallCurve(model)
                    viz.fit(X, y)
                    viz.score(X, y)
                    visualizations['pr_curve'] = self._fig_to_base64(viz.fig)
                    
                # Confusion matrix heatmap
                cm = confusion_matrix(y, model.predict(X))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                visualizations['confusion_matrix'] = self._fig_to_base64()
                
            else:  # regression
                # Residuals plot
                viz = ResidualsPlot(model)
                viz.fit(X, y)
                viz.score(X, y)
                visualizations['residuals'] = self._fig_to_base64(viz.fig)
                
                # Actual vs Predicted
                y_pred = model.predict(X)
                plt.figure(figsize=(10, 6))
                plt.scatter(y, y_pred, alpha=0.5)
                plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
                plt.xlabel('Actual')
                plt.ylabel('Predicted')
                plt.title('Actual vs Predicted')
                visualizations['actual_vs_predicted'] = self._fig_to_base64()
                
        except Exception as e:
            logging.error(f"Error generating visualizations: {str(e)}")
            
        return visualizations
        
    def explain_prediction(
        self,
        model: Any,
        X: np.ndarray,
        instance_index: int
    ) -> Dict[str, Any]:
        """Explain a specific prediction using SHAP and LIME"""
        explanations = {}
        
        try:
            # SHAP explanation
            if self.shap_explainer is not None:
                shap_values = self.shap_explainer.shap_values(X[instance_index:instance_index+1])
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                
                plt.figure()
                shap.summary_plot(
                    shap_values,
                    X[instance_index:instance_index+1],
                    feature_names=self.feature_names,
                    show=False
                )
                explanations['shap_summary'] = self._fig_to_base64()
                
                explanations['shap_values'] = {
                    self.feature_names[i]: float(shap_values[0][i])
                    for i in range(len(self.feature_names))
                }
                
            # LIME explanation
            if self.lime_explainer is not None:
                explanation = self.lime_explainer.explain_instance(
                    X[instance_index],
                    model.predict_proba if self.task_type == 'classification' else model.predict
                )
                
                explanations['lime_explanation'] = {
                    'feature_weights': explanation.as_list(),
                    'prediction': explanation.predict_proba if self.task_type == 'classification' else explanation.predict
                }
                
        except Exception as e:
            logging.error(f"Error generating explanations: {str(e)}")
            explanations['error'] = str(e)
            
        return explanations
        
    @staticmethod
    def _fig_to_base64(fig=None) -> str:
        """Convert matplotlib figure to base64 string"""
        if fig is None:
            fig = plt.gcf()
            
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        plt.close(fig)
        
        return img_str
