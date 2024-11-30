import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
from pathlib import Path
import time

@dataclass
class ModelComparison:
    """Data class for model comparison results"""
    model_name: str
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    training_time: float
    memory_usage: float
    feature_importance: Optional[Dict[str, float]] = None
    
class ModelComparer:
    def __init__(self):
        self.theme = {
            'background': '#ffffff',
            'text': '#262730',
            'primary': '#4CAF50',
            'secondary': '#45a049'
        }
        
    def compare_models(
        self,
        models: List[ModelComparison],
        metric_name: str = 'accuracy'
    ):
        """Compare multiple models across different aspects"""
        st.markdown("## 📊 Model Comparison")
        
        # Performance comparison
        self._compare_performance(models, metric_name)
        
        # Training time comparison
        self._compare_training_time(models)
        
        # Memory usage comparison
        self._compare_memory_usage(models)
        
        # Feature importance comparison
        self._compare_feature_importance(models)
        
        # Detailed comparison table
        self._show_comparison_table(models)
        
        # Export comparison
        self._export_comparison(models)
        
    def _compare_performance(
        self,
        models: List[ModelComparison],
        metric_name: str
    ):
        """Compare model performance"""
        st.markdown("### 📈 Performance Comparison")
        
        # Create performance dataframe
        performance_data = []
        for model in models:
            for metric, value in model.metrics.items():
                performance_data.append({
                    'Model': model.model_name,
                    'Metric': metric,
                    'Value': value
                })
                
        df = pd.DataFrame(performance_data)
        
        # Create interactive bar plot
        fig = px.bar(
            df[df['Metric'] == metric_name],
            x='Model',
            y='Value',
            title=f"{metric_name.title()} Comparison",
            color='Model'
        )
        
        fig.update_layout(
            plot_bgcolor='white',
            hoverlabel=dict(bgcolor="white"),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show other metrics
        if len(df['Metric'].unique()) > 1:
            selected_metric = st.selectbox(
                "Select another metric to compare",
                [m for m in df['Metric'].unique() if m != metric_name]
            )
            
            fig = px.bar(
                df[df['Metric'] == selected_metric],
                x='Model',
                y='Value',
                title=f"{selected_metric.title()} Comparison",
                color='Model'
            )
            
            fig.update_layout(
                plot_bgcolor='white',
                hoverlabel=dict(bgcolor="white"),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    def _compare_training_time(
        self,
        models: List[ModelComparison]
    ):
        """Compare model training times"""
        st.markdown("### ⏱️ Training Time Comparison")
        
        # Create training time dataframe
        df = pd.DataFrame([
            {
                'Model': model.model_name,
                'Training Time (s)': model.training_time
            }
            for model in models
        ])
        
        fig = px.bar(
            df,
            x='Model',
            y='Training Time (s)',
            title="Training Time Comparison",
            color='Model'
        )
        
        fig.update_layout(
            plot_bgcolor='white',
            hoverlabel=dict(bgcolor="white"),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def _compare_memory_usage(
        self,
        models: List[ModelComparison]
    ):
        """Compare model memory usage"""
        st.markdown("### 💾 Memory Usage Comparison")
        
        # Create memory usage dataframe
        df = pd.DataFrame([
            {
                'Model': model.model_name,
                'Memory Usage (MB)': model.memory_usage
            }
            for model in models
        ])
        
        fig = px.bar(
            df,
            x='Model',
            y='Memory Usage (MB)',
            title="Memory Usage Comparison",
            color='Model'
        )
        
        fig.update_layout(
            plot_bgcolor='white',
            hoverlabel=dict(bgcolor="white"),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def _compare_feature_importance(
        self,
        models: List[ModelComparison]
    ):
        """Compare feature importance across models"""
        st.markdown("### 🎯 Feature Importance Comparison")
        
        # Collect feature importance data
        feature_importance_data = []
        for model in models:
            if model.feature_importance:
                for feature, importance in model.feature_importance.items():
                    feature_importance_data.append({
                        'Model': model.model_name,
                        'Feature': feature,
                        'Importance': importance
                    })
                    
        if feature_importance_data:
            df = pd.DataFrame(feature_importance_data)
            
            # Create heatmap
            pivot_df = df.pivot(
                index='Feature',
                columns='Model',
                values='Importance'
            )
            
            fig = px.imshow(
                pivot_df,
                title="Feature Importance Heatmap",
                color_continuous_scale='RdBu'
            )
            
            fig.update_layout(
                plot_bgcolor='white',
                hoverlabel=dict(bgcolor="white"),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top features per model
            st.markdown("#### Top Features by Model")
            
            cols = st.columns(len(models))
            for col, model in zip(cols, models):
                if model.feature_importance:
                    with col:
                        st.markdown(f"**{model.model_name}**")
                        top_features = dict(
                            sorted(
                                model.feature_importance.items(),
                                key=lambda x: x[1],
                                reverse=True
                            )[:5]
                        )
                        for feature, importance in top_features.items():
                            st.markdown(f"- {feature}: {importance:.3f}")
                            
    def _show_comparison_table(
        self,
        models: List[ModelComparison]
    ):
        """Show detailed comparison table"""
        st.markdown("### 📋 Detailed Comparison")
        
        # Create comparison dataframe
        comparison_data = []
        for model in models:
            data = {
                'Model': model.model_name,
                'Training Time (s)': f"{model.training_time:.2f}",
                'Memory Usage (MB)': f"{model.memory_usage:.2f}"
            }
            data.update({
                k.title(): f"{v:.4f}"
                for k, v in model.metrics.items()
            })
            comparison_data.append(data)
            
        df = pd.DataFrame(comparison_data)
        st.dataframe(df)
        
        # Show model parameters
        with st.expander("🔧 Model Parameters", expanded=False):
            for model in models:
                st.markdown(f"**{model.model_name}**")
                st.json(model.parameters)
                
    def _export_comparison(
        self,
        models: List[ModelComparison]
    ):
        """Export comparison results"""
        st.markdown("### 💾 Export Comparison")
        
        if st.button("Export Comparison Results"):
            # Create export data
            export_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'models': [
                    {
                        'name': model.model_name,
                        'metrics': model.metrics,
                        'parameters': model.parameters,
                        'training_time': model.training_time,
                        'memory_usage': model.memory_usage,
                        'feature_importance': model.feature_importance
                    }
                    for model in models
                ]
            }
            
            # Convert to JSON
            json_str = json.dumps(export_data, indent=2)
            
            # Create download link
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="model_comparison.json">Download Comparison Results</a>'
            st.markdown(href, unsafe_allow_html=True)
            
    def save_comparison(
        self,
        models: List[ModelComparison],
        file_path: str
    ):
        """Save comparison results to file"""
        export_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'models': [
                {
                    'name': model.model_name,
                    'metrics': model.metrics,
                    'parameters': model.parameters,
                    'training_time': model.training_time,
                    'memory_usage': model.memory_usage,
                    'feature_importance': model.feature_importance
                }
                for model in models
            ]
        }
        
        Path(file_path).write_text(
            json.dumps(export_data, indent=2),
            encoding='utf-8'
        )
        
    def load_comparison(
        self,
        file_path: str
    ) -> List[ModelComparison]:
        """Load comparison results from file"""
        data = json.loads(
            Path(file_path).read_text(encoding='utf-8')
        )
        
        return [
            ModelComparison(
                model_name=model['name'],
                metrics=model['metrics'],
                parameters=model['parameters'],
                training_time=model['training_time'],
                memory_usage=model['memory_usage'],
                feature_importance=model['feature_importance']
            )
            for model in data['models']
        ]
