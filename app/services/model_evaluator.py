import numpy as np
from typing import Dict, Any, Optional, List, Union
import numpy.typing as npt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    explained_variance_score, mean_absolute_percentage_error,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.model_selection import learning_curve, validation_curve

class ModelEvaluator:
    """Handles model evaluation and performance metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics = {}
        self.plots = {}
        
    def evaluate_classification(self, y_true, y_pred):
        """
        Evaluate classification model performance.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            dict: Dictionary containing performance metrics
        """
        metrics = {
            # Core classification metrics
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
        }
        
        # Add ROC AUC if binary classification
        if len(np.unique(y_true)) == 2:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
            
        return metrics

    def evaluate_regression(self, y_true, y_pred):
        """
        Evaluate regression model performance.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            dict: Dictionary containing performance metrics
        """
        metrics = {
            # Core regression metrics
            'r2': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred)
        }
        
        # Add MAPE if no zero values in y_true
        if not np.any(y_true == 0):
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
            
        return metrics
        
    def evaluate_clustering(self, X, labels):
        """
        Evaluate clustering model performance.
        
        Args:
            X: Input features
            labels: Cluster labels
            
        Returns:
            dict: Dictionary containing clustering metrics
        """
        # Core clustering information
        n_clusters = len(np.unique(labels))
        metrics = {
            'n_clusters': n_clusters,
        }
        
        # Only calculate metrics if we have more than one cluster
        if n_clusters < 2:
            metrics['status'] = 'insufficient_clusters'
            metrics['message'] = 'Cannot calculate metrics with less than 2 clusters'
            return metrics
            
        # Core clustering metrics
        metrics.update({
            'silhouette': silhouette_score(X, labels),
            'calinski_harabasz': calinski_harabasz_score(X, labels),
            'davies_bouldin': davies_bouldin_score(X, labels)
        })
        
        metrics['status'] = 'success'
        return metrics
        
    def evaluate_dimensionality_reduction(self, X_original, X_reduced, model=None):
        """
        Evaluate dimensionality reduction model performance.
        
        Args:
            X_original: Original high-dimensional data
            X_reduced: Reduced low-dimensional data
            model: The dimensionality reduction model (optional)
            
        Returns:
            dict: Dictionary containing dimensionality reduction metrics
        """
        metrics = {
            # Core dimensionality metrics
            'original_dims': X_original.shape[1],
            'reduced_dims': X_reduced.shape[1],
            'reduction_ratio': X_reduced.shape[1] / X_original.shape[1]
        }
        
        # Model-specific metrics
        if hasattr(model, 'explained_variance_ratio_'):
            metrics['explained_variance'] = np.sum(model.explained_variance_ratio_)
            
        if hasattr(model, 'inverse_transform'):
            X_reconstructed = model.inverse_transform(X_reduced)
            metrics['reconstruction_error'] = np.mean(np.square(X_original - X_reconstructed))
            
        return metrics
        
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Create labels
        labels = [f'Class {i}' for i in range(cm.shape[0])]
        
        # Create heatmap
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Blues',
            showscale=True
        )
        
        # Update layout
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual'
        )
        
        return fig
        
    def plot_regression_scatter(self, y_true, y_pred, title="Actual vs Predicted Values"):
        """Create a scatter plot of actual vs predicted values for regression
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            title: Plot title
            
        Returns:
            plotly.graph_objects.Figure: Scatter plot figure
        """
        try:
            self.logger.debug("Creating regression scatter plot")
            
            # Create scatter plot
            fig = go.Figure()
            
            # Add scatter plot of actual vs predicted
            fig.add_trace(
                go.Scatter(
                    x=y_true,
                    y=y_pred,
                    mode='markers',
                    name='Predictions',
                    marker=dict(
                        size=8,
                        opacity=0.6,
                        color='blue'
                    )
                )
            )
            
            # Add perfect prediction line
            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(
                        color='red',
                        dash='dash'
                    )
                )
            )
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Actual Values",
                yaxis_title="Predicted Values",
                showlegend=True,
                template='plotly_white',
                hovermode='closest'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error in plot_regression_scatter: {str(e)}")
            self.logger.exception(e)
            raise
            
    def plot_regression_residuals(self, y_true, y_pred, title="Residual Plot"):
        """Create a residual plot for regression
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            title: Plot title
            
        Returns:
            plotly.graph_objects.Figure: Residual plot figure
        """
        try:
            self.logger.debug("Creating residual plot")
            
            # Calculate residuals
            residuals = y_true - y_pred
            
            # Create figure
            fig = go.Figure()
            
            # Add scatter plot of predicted vs residuals
            fig.add_trace(
                go.Scatter(
                    x=y_pred,
                    y=residuals,
                    mode='markers',
                    name='Residuals',
                    marker=dict(
                        size=8,
                        opacity=0.6,
                        color='blue'
                    )
                )
            )
            
            # Add zero line
            fig.add_hline(
                y=0,
                line_dash="dash",
                line_color="red",
                name="Zero Line"
            )
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Predicted Values",
                yaxis_title="Residuals",
                showlegend=True,
                template='plotly_white',
                hovermode='closest'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error in plot_regression_residuals: {str(e)}")
            self.logger.exception(e)
            raise
            
    def plot_feature_importance(self, model, feature_names, title="Feature Importance"):
        """Create a bar plot of feature importance
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            title: Plot title
            
        Returns:
            plotly.graph_objects.Figure: Feature importance plot figure
        """
        try:
            self.logger.debug("Creating feature importance plot")
            
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            else:
                raise ValueError("Model does not have feature_importances_ attribute")
                
            # Sort features by importance
            indices = np.argsort(importance)[::-1]
            sorted_features = [feature_names[i] for i in indices]
            sorted_importance = importance[indices]
            
            # Create figure
            fig = go.Figure()
            
            # Add bar plot
            fig.add_trace(
                go.Bar(
                    x=sorted_importance,
                    y=sorted_features,
                    orientation='h',
                    marker_color='blue'
                )
            )
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Importance",
                yaxis_title="Features",
                showlegend=False,
                template='plotly_white',
                height=max(400, len(feature_names) * 20)  # Adjust height based on number of features
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error in plot_feature_importance: {str(e)}")
            self.logger.exception(e)
            raise
            
    def plot_residuals(self, y_true, y_pred):
        """Plot residuals"""
        residuals = y_true - y_pred
        
        fig = px.scatter(
            x=y_pred,
            y=residuals,
            labels={'x': 'Predicted Values', 'y': 'Residuals'},
            title='Residual Plot'
        )
        
        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        
        return fig
        
    def plot_feature_importance(self, model, feature_names):
        """Plot feature importance"""
        importances = model.feature_importances_
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        importance_df = importance_df.sort_values('Importance', ascending=True)
        
        # Create bar plot
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance'
        )
        
        return fig
        
    def get_classification_report(self, y_true, y_pred):
        """Get detailed classification report"""
        from sklearn.metrics import classification_report
        return classification_report(y_true, y_pred)
        
    def get_regression_report(self, y_true, y_pred):
        """Get detailed regression report"""
        metrics = self.evaluate_regression(y_true, y_pred)
        
        report = [
            "Regression Statistics:",
            f"R² Score: {metrics['r2']:.3f}",
            f"Mean Squared Error: {metrics['mse']:.3f}",
            f"Root Mean Squared Error: {metrics['rmse']:.3f}",
            f"Mean Absolute Error: {metrics['mae']:.3f}",
            f"Mean Absolute Percentage Error: {metrics.get('mape', 'N/A'):.3f}",
            "",
            "Additional Statistics:",
            f"Mean of True Values: {np.mean(y_true):.3f}",
            f"Std of True Values: {np.std(y_true):.3f}",
            f"Mean of Predictions: {np.mean(y_pred):.3f}",
            f"Std of Predictions: {np.std(y_pred):.3f}"
        ]
        
        return "\n".join(report)
        
    def get_clustering_report(self, X, labels):
        """Get detailed clustering report"""
        from sklearn.metrics import (
            silhouette_score,
            calinski_harabasz_score,
            davies_bouldin_score
        )
        
        report = [
            "Clustering Statistics:",
            f"Number of Clusters: {len(np.unique(labels))}",
            f"Silhouette Score: {silhouette_score(X, labels):.3f}",
            f"Calinski-Harabasz Score: {calinski_harabasz_score(X, labels):.3f}",
            f"Davies-Bouldin Score: {davies_bouldin_score(X, labels):.3f}",
            "",
            "Cluster Distribution:",
        ]
        
        # Add cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        for cluster, count in zip(unique, counts):
            report.append(f"Cluster {cluster}: {count} samples ({count/len(labels)*100:.1f}%)")
            
        return "\n".join(report)
        
    def get_dim_reduction_report(self, X_original, X_transformed):
        """Get detailed dimensionality reduction report"""
        report = [
            "Dimensionality Reduction Statistics:",
            f"Original Dimensions: {X_original.shape[1]}",
            f"Reduced Dimensions: {X_transformed.shape[1]}",
            f"Dimension Reduction Ratio: {X_transformed.shape[1]/X_original.shape[1]:.2f}",
            "",
            "Component Statistics:"
        ]
        
        # Add explained variance per component if available
        if hasattr(X_transformed, 'explained_variance_ratio_'):
            for i, var in enumerate(X_transformed.explained_variance_ratio_):
                report.append(f"Component {i+1}: {var*100:.1f}% variance explained")
                
        return "\n".join(report)
        
    def evaluate(self, X, y, model=None, task_type=None, feature_names=None):
        """
        Evaluate model performance
        Args:
            X: Features
            y: Target variable or true labels
            model: Trained model (optional)
            task_type: Type of task ('classification', 'regression', 'clustering', 'dim_reduction')
            feature_names: List of feature names for feature importance plots
        Returns:
            Dictionary containing evaluation metrics and plots
        """
        try:
            self.logger.info(f"Starting model evaluation for task type: {task_type}")
            results = {
                'metrics': {},
                'plots': {},
                'feature_importance': None,
                'detailed_report': None
            }
            
            # Automatically detect task type if not provided
            if task_type is None and model is not None:
                if hasattr(model, 'predict_proba'):
                    task_type = 'classification'
                elif hasattr(model, 'predict'):
                    task_type = 'regression'
                elif hasattr(model, 'labels_'):
                    task_type = 'clustering'
                elif hasattr(model, 'transform'):
                    task_type = 'dim_reduction'
                    
            # Get predictions if model is provided
            if model is not None:
                if task_type in ['classification', 'regression']:
                    y_pred = model.predict(X)
                elif task_type == 'clustering':
                    y_pred = model.labels_
                elif task_type == 'dim_reduction':
                    X_transformed = model.transform(X)
                    
            # Evaluate based on task type
            if task_type == 'classification':
                # Classification metrics
                results['metrics'] = self.evaluate_classification(y, y_pred)
                results['detailed_report'] = self.get_classification_report(y, y_pred)
                
                # Classification plots
                results['plots']['confusion_matrix'] = self.plot_confusion_matrix(y, y_pred)
                if len(np.unique(y)) == 2:  # Binary classification
                    y_prob = model.predict_proba(X)[:, 1]
                    results['plots']['roc_curve'] = self.plot_roc_curve(y, y_prob)
                    results['plots']['pr_curve'] = self.plot_precision_recall_curve(y, y_prob)
                    
            elif task_type == 'regression':
                # Regression metrics
                results['metrics'] = self.evaluate_regression(y, y_pred)
                results['detailed_report'] = self.get_regression_report(y, y_pred)
                
                # Regression plots
                results['plots']['scatter'] = self.plot_regression_scatter(y, y_pred)
                results['plots']['residuals'] = self.plot_regression_residuals(y, y_pred)
                
            elif task_type == 'clustering':
                # Clustering metrics
                results['metrics'] = self.evaluate_clustering(X, y_pred)
                results['detailed_report'] = self.get_clustering_report(X, y_pred)
                
                # Clustering plots
                if X.shape[1] == 2:  # 2D data
                    results['plots']['clusters_2d'] = self.plot_clusters_2d(X, y_pred)
                elif X.shape[1] == 3:  # 3D data
                    results['plots']['clusters_3d'] = self.plot_clusters_3d(X, y_pred)
                    
            elif task_type == 'dim_reduction':
                # Dimensionality reduction metrics
                results['metrics'] = self.evaluate_dimensionality_reduction(X, X_transformed, model)
                results['detailed_report'] = self.get_dim_reduction_report(X, X_transformed)
                
                # Dimensionality reduction plots
                if X_transformed.shape[1] in [2, 3]:
                    results['plots']['transformed_data'] = self.plot_transformed_data(X_transformed, y)
                    
            # Feature importance (if available)
            if model is not None and feature_names is not None:
                if hasattr(model, 'feature_importances_'):
                    results['feature_importance'] = model.feature_importances_
                    results['plots']['feature_importance'] = self.plot_feature_importance(
                        model, feature_names
                    )
                elif hasattr(model, 'coef_'):
                    if len(model.coef_.shape) > 1:
                        results['feature_importance'] = np.abs(model.coef_).mean(axis=0)
                    else:
                        results['feature_importance'] = np.abs(model.coef_)
                    results['plots']['feature_importance'] = self.plot_feature_importance(
                        model, feature_names, coef_based=True
                    )
                    
            self.logger.info("Model evaluation completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in model evaluation: {str(e)}")
            self.logger.exception(e)
            raise
            
    def save_results(self, save_dir: str) -> None:
        """Save evaluation results to disk."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_file = save_dir / 'metrics.joblib'
        joblib.dump(self.metrics, metrics_file)
        
        # Save plots
        for name, fig in self.plots.items():
            plot_file = save_dir / f'{name}.png'
            fig.savefig(plot_file)
            plt.close(fig)
            
    def load_results(self, save_dir: str) -> None:
        """Load evaluation results from disk."""
        save_dir = Path(save_dir)
        
        # Load metrics
        metrics_file = save_dir / 'metrics.joblib'
        if metrics_file.exists():
            self.metrics = joblib.load(metrics_file)
            
        # Load plots
        for plot_file in save_dir.glob('*.png'):
            name = plot_file.stem
            self.plots[name] = plt.imread(str(plot_file))

    def plot_learning_curves(self, model, X, y, cv=5):
        """Plot learning curves to analyze model performance vs training size
        
        Args:
            model: Trained model
            X: Features
            y: Target variable
            cv: Number of cross-validation folds
            
        Returns:
            plotly.graph_objects.Figure: Learning curves plot
        """
        try:
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_sizes, train_scores, val_scores = learning_curve(
                model, X, y, 
                train_sizes=train_sizes,
                cv=cv, n_jobs=-1,
                scoring='r2' if hasattr(model, 'predict') else 'accuracy'
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            fig = go.Figure()
            
            # Training scores
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=train_mean,
                name='Training Score',
                mode='lines+markers',
                line=dict(color='blue'),
                error_y=dict(
                    type='data',
                    array=train_std,
                    visible=True
                )
            ))
            
            # Validation scores
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=val_mean,
                name='Cross-validation Score',
                mode='lines+markers',
                line=dict(color='red'),
                error_y=dict(
                    type='data',
                    array=val_std,
                    visible=True
                )
            ))
            
            fig.update_layout(
                title='Learning Curves',
                xaxis_title='Training Examples',
                yaxis_title='Score',
                hovermode='x unified',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting learning curves: {str(e)}")
            self.logger.exception(e)
            raise
            
    def plot_validation_curve(self, model, X, y, param_name, param_range, cv=5):
        """Plot validation curve for a specific hyperparameter
        
        Args:
            model: Model instance
            X: Features
            y: Target variable
            param_name: Name of the parameter to validate
            param_range: Range of parameter values to try
            cv: Number of cross-validation folds
            
        Returns:
            plotly.graph_objects.Figure: Validation curve plot
        """
        try:
            train_scores, val_scores = validation_curve(
                model, X, y,
                param_name=param_name,
                param_range=param_range,
                cv=cv, n_jobs=-1,
                scoring='r2' if hasattr(model, 'predict') else 'accuracy'
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            fig = go.Figure()
            
            # Training scores
            fig.add_trace(go.Scatter(
                x=param_range,
                y=train_mean,
                name='Training Score',
                mode='lines+markers',
                line=dict(color='blue'),
                error_y=dict(
                    type='data',
                    array=train_std,
                    visible=True
                )
            ))
            
            # Validation scores
            fig.add_trace(go.Scatter(
                x=param_range,
                y=val_mean,
                name='Cross-validation Score',
                mode='lines+markers',
                line=dict(color='red'),
                error_y=dict(
                    type='data',
                    array=val_std,
                    visible=True
                )
            ))
            
            fig.update_layout(
                title=f'Validation Curve ({param_name})',
                xaxis_title=param_name,
                yaxis_title='Score',
                hovermode='x unified',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting validation curve: {str(e)}")
            self.logger.exception(e)
            raise
