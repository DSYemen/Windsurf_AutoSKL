from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass

@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    class_report: Dict
    roc_auc: Optional[float] = None
    fpr: Optional[np.ndarray] = None
    tpr: Optional[np.ndarray] = None
    
@dataclass
class RegressionMetrics:
    mse: float
    rmse: float
    mae: float
    r2: float
    explained_variance: float
    residuals: np.ndarray

class ModelEvaluator:
    @staticmethod
    def evaluate_classification(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> ClassificationMetrics:
        """Evaluate classification model performance"""
        metrics = ClassificationMetrics(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, average='weighted'),
            recall=recall_score(y_true, y_pred, average='weighted'),
            f1=f1_score(y_true, y_pred, average='weighted'),
            confusion_matrix=confusion_matrix(y_true, y_pred),
            class_report=classification_report(y_true, y_pred, output_dict=True)
        )
        
        if y_prob is not None and y_prob.shape[1] == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            metrics.roc_auc = auc(fpr, tpr)
            metrics.fpr = fpr
            metrics.tpr = tpr
            
        return metrics
    
    @staticmethod
    def evaluate_regression(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> RegressionMetrics:
        """Evaluate regression model performance"""
        residuals = y_true - y_pred
        return RegressionMetrics(
            mse=mean_squared_error(y_true, y_pred),
            rmse=np.sqrt(mean_squared_error(y_true, y_pred)),
            mae=mean_absolute_error(y_true, y_pred),
            r2=r2_score(y_true, y_pred),
            explained_variance=1 - (np.var(residuals) / np.var(y_true)),
            residuals=residuals
        )
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, classes: List[str]) -> go.Figure:
        """Create interactive confusion matrix plot"""
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=classes,
            y=classes,
            colorscale='Viridis',
            text=cm.astype(str),
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            xaxis={'side': 'bottom'}
        )
        
        return fig
    
    @staticmethod
    def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float) -> go.Figure:
        """Create ROC curve plot"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC curve (AUC = {roc_auc:.2f})'
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash'),
            name='Random'
        ))
        
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=700, height=500
        )
        
        return fig
    
    @staticmethod
    def plot_residuals(residuals: np.ndarray, y_pred: np.ndarray) -> go.Figure:
        """Create residuals plot"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            marker=dict(
                size=8,
                color='blue',
                opacity=0.6
            ),
            name='Residuals'
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title='Residuals vs Predicted Values',
            xaxis_title='Predicted Values',
            yaxis_title='Residuals',
            width=700, height=500
        )
        
        return fig
    
    @staticmethod
    def plot_feature_importance(
        feature_names: List[str],
        importance_values: np.ndarray
    ) -> go.Figure:
        """Create feature importance plot"""
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_values
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
            width=800,
            height=max(400, len(feature_names) * 20)
        )
        
        return fig
