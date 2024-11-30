import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class ModelVisualizer:
    def __init__(self):
        self.theme = {
            'background': '#ffffff',
            'text': '#262730',
            'primary': '#4CAF50',
            'secondary': '#45a049'
        }
        
    def visualize_decision_boundary(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None
    ):
        """Visualize decision boundary for 2D data"""
        if X.shape[1] != 2:
            st.warning("Decision boundary visualization requires 2D data")
            return
            
        # Create mesh grid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.1),
            np.arange(y_min, y_max, 0.1)
        )
        
        # Make predictions
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Create figure
        fig = go.Figure()
        
        # Add decision boundary
        fig.add_trace(go.Contour(
            x=np.arange(x_min, x_max, 0.1),
            y=np.arange(y_min, y_max, 0.1),
            z=Z,
            colorscale='RdBu',
            showscale=False,
            opacity=0.3
        ))
        
        # Add scatter plot of actual points
        for i in np.unique(y):
            mask = y == i
            name = class_names[i] if class_names else f"Class {i}"
            fig.add_trace(go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode='markers',
                name=name,
                marker=dict(size=8)
            ))
            
        # Update layout
        fig.update_layout(
            title="Decision Boundary Visualization",
            xaxis_title=feature_names[0] if feature_names else "Feature 1",
            yaxis_title=feature_names[1] if feature_names else "Feature 2",
            plot_bgcolor='white',
            hoverlabel=dict(bgcolor="white"),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def visualize_feature_interactions(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ):
        """Visualize feature interactions using SHAP"""
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
                
            # SHAP summary plot
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(
                shap_values,
                X,
                feature_names=feature_names,
                show=False
            )
            plt.tight_layout()
            st.pyplot(fig)
            
            # SHAP interaction plot
            st.markdown("### Feature Interactions")
            
            selected_feature = st.selectbox(
                "Select feature to analyze interactions",
                feature_names if feature_names else [f"Feature {i}" for i in range(X.shape[1])]
            )
            
            feature_idx = (
                feature_names.index(selected_feature)
                if feature_names
                else int(selected_feature.split()[-1])
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.dependence_plot(
                feature_idx,
                shap_values,
                X,
                feature_names=feature_names,
                show=False
            )
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error generating SHAP plots: {str(e)}")
            
    def visualize_feature_importance(
        self,
        importance_dict: Dict[str, float],
        top_n: int = 10
    ):
        """Visualize feature importance with detailed insights"""
        # Sort features by importance
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Take top N features
        top_features = sorted_features[:top_n]
        
        # Create figure
        fig = go.Figure()
        
        # Add horizontal bar chart
        fig.add_trace(go.Bar(
            y=[f[0] for f in top_features],
            x=[f[1] for f in top_features],
            orientation='h',
            marker_color=self.theme['primary']
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Top {top_n} Most Important Features",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            plot_bgcolor='white',
            hoverlabel=dict(bgcolor="white"),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional insights
        st.markdown("### 💡 Feature Importance Insights")
        
        # Calculate statistics
        total_importance = sum(importance_dict.values())
        cumulative_importance = sum(f[1] for f in top_features)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Top Features Contribution",
                f"{cumulative_importance/total_importance*100:.1f}%"
            )
            
        with col2:
            st.metric(
                "Number of Important Features",
                len([f for f in importance_dict.values() if f > 0.01])
            )
            
    def visualize_manifold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = 'tsne',
        perplexity: int = 30,
        n_neighbors: int = 15
    ):
        """Visualize high-dimensional data using manifold learning"""
        # Standardize data
        X_scaled = StandardScaler().fit_transform(X)
        
        # Perform dimensionality reduction
        if method.lower() == 'tsne':
            reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=42
            )
        else:  # UMAP
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                random_state=42
            )
            
        X_embedded = reducer.fit_transform(X_scaled)
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot for each class
        for i in np.unique(y):
            mask = y == i
            fig.add_trace(go.Scatter(
                x=X_embedded[mask, 0],
                y=X_embedded[mask, 1],
                mode='markers',
                name=f"Class {i}",
                marker=dict(size=8)
            ))
            
        # Update layout
        fig.update_layout(
            title=f"{method.upper()} Visualization",
            xaxis_title=f"{method.upper()} 1",
            yaxis_title=f"{method.upper()} 2",
            plot_bgcolor='white',
            hoverlabel=dict(bgcolor="white"),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def visualize_prediction_confidence(
        self,
        model: Any,
        X: np.ndarray,
        y_true: np.ndarray,
        class_names: Optional[List[str]] = None
    ):
        """Visualize model prediction confidence"""
        try:
            # Get prediction probabilities
            y_prob = model.predict_proba(X)
            
            # Get predicted classes
            y_pred = model.predict(X)
            
            # Calculate confidence (max probability)
            confidence = np.max(y_prob, axis=1)
            
            # Create figure
            fig = go.Figure()
            
            # Add histogram for correct and incorrect predictions
            correct = y_pred == y_true
            
            fig.add_trace(go.Histogram(
                x=confidence[correct],
                name="Correct Predictions",
                marker_color=self.theme['primary'],
                opacity=0.7
            ))
            
            fig.add_trace(go.Histogram(
                x=confidence[~correct],
                name="Incorrect Predictions",
                marker_color='red',
                opacity=0.7
            ))
            
            # Update layout
            fig.update_layout(
                title="Prediction Confidence Distribution",
                xaxis_title="Confidence Score",
                yaxis_title="Count",
                barmode='overlay',
                plot_bgcolor='white',
                hoverlabel=dict(bgcolor="white"),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional metrics
            st.markdown("### 📊 Confidence Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Average Confidence",
                    f"{confidence.mean():.3f}"
                )
                
            with col2:
                st.metric(
                    "High Confidence Predictions",
                    f"{(confidence > 0.9).mean()*100:.1f}%"
                )
                
            with col3:
                st.metric(
                    "Low Confidence Predictions",
                    f"{(confidence < 0.5).mean()*100:.1f}%"
                )
                
        except Exception as e:
            st.error(f"Error generating confidence visualization: {str(e)}")
            
    def visualize_learning_curve(
        self,
        train_sizes: np.ndarray,
        train_scores: np.ndarray,
        val_scores: np.ndarray
    ):
        """Visualize learning curves with confidence intervals"""
        fig = go.Figure()
        
        # Calculate means and standard deviations
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Add training score
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_mean,
            name="Training Score",
            line=dict(color=self.theme['primary']),
            mode='lines+markers'
        ))
        
        # Add training score confidence interval
        fig.add_trace(go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]),
            y=np.concatenate([
                train_mean + train_std,
                (train_mean - train_std)[::-1]
            ]),
            fill='toself',
            fillcolor=f"rgba{tuple(int(self.theme['primary'][i:i+2], 16) for i in (1, 3, 5)) + (0.2,)}",
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name="Training Confidence"
        ))
        
        # Add validation score
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=val_mean,
            name="Validation Score",
            line=dict(color=self.theme['secondary']),
            mode='lines+markers'
        ))
        
        # Add validation score confidence interval
        fig.add_trace(go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]),
            y=np.concatenate([
                val_mean + val_std,
                (val_mean - val_std)[::-1]
            ]),
            fill='toself',
            fillcolor=f"rgba{tuple(int(self.theme['secondary'][i:i+2], 16) for i in (1, 3, 5)) + (0.2,)}",
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name="Validation Confidence"
        ))
        
        # Update layout
        fig.update_layout(
            title="Learning Curves",
            xaxis_title="Training Examples",
            yaxis_title="Score",
            plot_bgcolor='white',
            hoverlabel=dict(bgcolor="white"),
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(
                yanchor="bottom",
                y=0.01,
                xanchor="right",
                x=0.99
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional insights
        st.markdown("### 💡 Learning Curve Insights")
        
        # Calculate metrics
        final_train_score = train_mean[-1]
        final_val_score = val_mean[-1]
        score_gap = final_train_score - final_val_score
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Final Training Score",
                f"{final_train_score:.3f}"
            )
            
        with col2:
            st.metric(
                "Final Validation Score",
                f"{final_val_score:.3f}"
            )
            
        with col3:
            st.metric(
                "Generalization Gap",
                f"{score_gap:.3f}",
                delta="-" if score_gap > 0.1 else "+"
            )
            
        # Provide recommendations
        if score_gap > 0.1:
            st.warning("""
                ⚠️ High generalization gap detected. Consider:
                - Adding regularization
                - Reducing model complexity
                - Collecting more training data
                - Using cross-validation
            """)
        else:
            st.success("""
                ✅ Good generalization performance. Model shows:
                - Balanced training/validation scores
                - Stable learning curve
                - Low risk of overfitting
            """)
