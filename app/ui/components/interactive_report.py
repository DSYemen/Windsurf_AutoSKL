import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import base64
from datetime import datetime

class InteractiveReport:
    def __init__(self):
        self.theme = {
            'background': '#ffffff',
            'text': '#262730',
            'primary': '#4CAF50',
            'secondary': '#45a049'
        }
        
    def create_header(self, title: str, subtitle: Optional[str] = None):
        """Create a styled header section"""
        st.markdown(f"""
            <h1 style='color: {self.theme["text"]}; margin-bottom: 0;'>{title}</h1>
            {f"<p style='color: {self.theme['text']}; font-size: 1.2em;'>{subtitle}</p>" if subtitle else ""}
        """, unsafe_allow_html=True)
        
    def create_metrics_dashboard(self, metrics: Dict[str, float]):
        """Create a dashboard of key metrics"""
        cols = st.columns(len(metrics))
        for col, (name, value) in zip(cols, metrics.items()):
            with col:
                st.metric(
                    label=name.replace('_', ' ').title(),
                    value=f"{value:.4f}" if isinstance(value, float) else value
                )
                
    def plot_feature_importance(
        self,
        importance_dict: Dict[str, float],
        title: str = "Feature Importance"
    ):
        """Create an interactive feature importance plot"""
        df = pd.DataFrame({
            'Feature': list(importance_dict.keys()),
            'Importance': list(importance_dict.values())
        }).sort_values('Importance', ascending=True)
        
        fig = go.Figure(go.Bar(
            x=df['Importance'],
            y=df['Feature'],
            orientation='h',
            marker_color=self.theme['primary']
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            plot_bgcolor='white',
            hoverlabel=dict(bgcolor="white"),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: Optional[List[str]] = None
    ):
        """Create an interactive confusion matrix plot"""
        if labels is None:
            labels = [f"Class {i}" for i in range(len(cm))]
            
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='RdBu',
            text=cm.astype(str),
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def plot_learning_curves(
        self,
        train_sizes: np.ndarray,
        train_scores: np.ndarray,
        val_scores: np.ndarray
    ):
        """Create an interactive learning curves plot"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_scores.mean(axis=1),
            name="Training Score",
            line=dict(color=self.theme['primary']),
            fill='tonexty'
        ))
        
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=val_scores.mean(axis=1),
            name="Validation Score",
            line=dict(color=self.theme['secondary']),
            fill='tonexty'
        ))
        
        fig.update_layout(
            title="Learning Curves",
            xaxis_title="Training Examples",
            yaxis_title="Score",
            plot_bgcolor='white',
            hoverlabel=dict(bgcolor="white"),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ):
        """Create an interactive residuals plot"""
        residuals = y_true - y_pred
        
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(
                color=self.theme['primary'],
                size=8,
                opacity=0.6
            )
        ))
        
        # Zero line
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="red",
            annotation_text="Zero Line"
        )
        
        fig.update_layout(
            title="Residuals Plot",
            xaxis_title="Predicted Values",
            yaxis_title="Residuals",
            plot_bgcolor='white',
            hoverlabel=dict(bgcolor="white"),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def create_download_button(
        self,
        data: Any,
        file_name: str,
        button_text: str = "Download Data"
    ):
        """Create a download button for data"""
        if isinstance(data, pd.DataFrame):
            data = data.to_csv(index=False)
        elif isinstance(data, dict):
            data = pd.DataFrame(data).to_csv(index=False)
            
        b64 = base64.b64encode(data.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}" class="download-button">{button_text}</a>'
        st.markdown(href, unsafe_allow_html=True)
        
    def create_model_card(
        self,
        model_info: Dict[str, Any]
    ):
        """Create an expandable model card with detailed information"""
        with st.expander("📋 Model Card", expanded=False):
            st.markdown(f"""
                ### Model Information
                - **Name**: {model_info.get('name', 'N/A')}
                - **Type**: {model_info.get('type', 'N/A')}
                - **Version**: {model_info.get('version', 'N/A')}
                - **Created**: {model_info.get('created', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}
                
                ### Performance Metrics
                {self._format_metrics(model_info.get('metrics', {}))}
                
                ### Model Parameters
                ```python
                {model_info.get('parameters', {})}
                ```
                
                ### Feature Information
                - **Input Features**: {len(model_info.get('features', []))}
                - **Target**: {model_info.get('target', 'N/A')}
                
                ### Usage Notes
                {model_info.get('usage_notes', 'No usage notes available.')}
            """)
            
    @staticmethod
    def _format_metrics(metrics: Dict[str, float]) -> str:
        """Format metrics for markdown display"""
        return "\n".join([
            f"- **{k.replace('_', ' ').title()}**: {v:.4f}"
            for k, v in metrics.items()
        ])
        
    def create_feature_analysis(
        self,
        feature_data: pd.DataFrame,
        feature_name: str
    ):
        """Create an interactive feature analysis section"""
        st.subheader(f"📊 Analysis of {feature_name}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Basic statistics
            stats = feature_data[feature_name].describe()
            st.markdown("### Basic Statistics")
            st.dataframe(stats)
            
        with col2:
            # Distribution plot
            if feature_data[feature_name].dtype in ['int64', 'float64']:
                fig = px.histogram(
                    feature_data,
                    x=feature_name,
                    nbins=30,
                    title=f"Distribution of {feature_name}"
                )
            else:
                fig = px.bar(
                    feature_data[feature_name].value_counts(),
                    title=f"Distribution of {feature_name}"
                )
                
            fig.update_layout(
                plot_bgcolor='white',
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        # Additional insights
        st.markdown("### 💡 Insights")
        
        # Missing values
        missing = feature_data[feature_name].isnull().sum()
        if missing > 0:
            st.warning(f"⚠️ Contains {missing} missing values ({missing/len(feature_data)*100:.2f}%)")
            
        # Unique values
        unique = feature_data[feature_name].nunique()
        st.info(f"🔍 Contains {unique} unique values")
        
        # Outliers for numerical features
        if feature_data[feature_name].dtype in ['int64', 'float64']:
            Q1 = feature_data[feature_name].quantile(0.25)
            Q3 = feature_data[feature_name].quantile(0.75)
            IQR = Q3 - Q1
            outliers = feature_data[
                (feature_data[feature_name] < (Q1 - 1.5 * IQR)) |
                (feature_data[feature_name] > (Q3 + 1.5 * IQR))
            ]
            if len(outliers) > 0:
                st.warning(f"⚠️ Contains {len(outliers)} potential outliers")
