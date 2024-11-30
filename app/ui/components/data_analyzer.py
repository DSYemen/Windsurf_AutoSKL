import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple
import base64
from io import BytesIO

class DataAnalyzer:
    """Interactive data analysis component for AutoSKL"""
    
    def __init__(self):
        self.theme = {
            'background': '#ffffff',
            'text': '#262730',
            'primary': '#4CAF50',
            'secondary': '#45a049'
        }
        
    def analyze_dataset(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Dict:
        """Perform comprehensive dataset analysis"""
        analysis = {
            'basic_info': self._get_basic_info(data),
            'missing_values': self._analyze_missing_values(data),
            'numerical_analysis': self._analyze_numerical_features(data),
            'categorical_analysis': self._analyze_categorical_features(data),
            'correlations': self._analyze_correlations(data)
        }
        
        if target_column:
            analysis['target_analysis'] = self._analyze_target(data, target_column)
            
        return analysis
    
    def show_analysis(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None
    ):
        """Display interactive data analysis in Streamlit"""
        analysis = self.analyze_dataset(data, target_column)
        
        # Basic Information
        with st.expander("📊 Dataset Overview", expanded=True):
            self._display_basic_info(analysis['basic_info'])
        
        # Missing Values Analysis
        with st.expander("🔍 Missing Values Analysis"):
            self._display_missing_values(analysis['missing_values'])
        
        # Numerical Features Analysis
        with st.expander("📈 Numerical Features Analysis"):
            self._display_numerical_analysis(
                data,
                analysis['numerical_analysis']
            )
        
        # Categorical Features Analysis
        with st.expander("📊 Categorical Features Analysis"):
            self._display_categorical_analysis(
                data,
                analysis['categorical_analysis']
            )
        
        # Correlation Analysis
        with st.expander("🔗 Correlation Analysis"):
            self._display_correlations(analysis['correlations'])
        
        # Target Analysis
        if target_column and 'target_analysis' in analysis:
            with st.expander("🎯 Target Variable Analysis"):
                self._display_target_analysis(
                    data,
                    target_column,
                    analysis['target_analysis']
                )
    
    def _get_basic_info(self, data: pd.DataFrame) -> Dict:
        """Get basic dataset information"""
        return {
            'shape': data.shape,
            'memory_usage': data.memory_usage(deep=True).sum(),
            'duplicates': data.duplicated().sum(),
            'dtypes': data.dtypes.value_counts().to_dict(),
            'columns': {
                col: str(dtype)
                for col, dtype in data.dtypes.items()
            }
        }
    
    def _analyze_missing_values(self, data: pd.DataFrame) -> Dict:
        """Analyze missing values in dataset"""
        missing = data.isnull().sum()
        return {
            'total_missing': missing.sum(),
            'missing_by_column': missing[missing > 0].to_dict(),
            'missing_percentages': (missing / len(data) * 100).to_dict()
        }
    
    def _analyze_numerical_features(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Dict]:
        """Analyze numerical features"""
        numerical_cols = data.select_dtypes(
            include=['int64', 'float64']
        ).columns
        
        return {
            col: {
                'mean': data[col].mean(),
                'median': data[col].median(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'skew': data[col].skew(),
                'kurtosis': data[col].kurtosis()
            }
            for col in numerical_cols
        }
    
    def _analyze_categorical_features(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Dict]:
        """Analyze categorical features"""
        categorical_cols = data.select_dtypes(
            include=['object', 'category']
        ).columns
        
        return {
            col: {
                'unique_values': data[col].nunique(),
                'value_counts': data[col].value_counts().to_dict()
            }
            for col in categorical_cols
        }
    
    def _analyze_correlations(self, data: pd.DataFrame) -> Dict:
        """Analyze feature correlations"""
        numerical_data = data.select_dtypes(
            include=['int64', 'float64']
        )
        
        if len(numerical_data.columns) > 1:
            correlations = numerical_data.corr()
            return {
                'correlation_matrix': correlations.to_dict(),
                'high_correlations': self._get_high_correlations(correlations)
            }
        return {}
    
    def _analyze_target(
        self,
        data: pd.DataFrame,
        target_column: str
    ) -> Dict:
        """Analyze target variable"""
        target_data = data[target_column]
        
        if target_data.dtype in ['int64', 'float64']:
            return {
                'type': 'numerical',
                'stats': {
                    'mean': target_data.mean(),
                    'median': target_data.median(),
                    'std': target_data.std(),
                    'min': target_data.min(),
                    'max': target_data.max()
                }
            }
        else:
            return {
                'type': 'categorical',
                'stats': {
                    'unique_values': target_data.nunique(),
                    'value_counts': target_data.value_counts().to_dict(),
                    'class_balance': (
                        target_data.value_counts() / len(target_data)
                    ).to_dict()
                }
            }
    
    def _display_basic_info(self, info: Dict):
        """Display basic dataset information"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Number of Rows",
                info['shape'][0]
            )
        with col2:
            st.metric(
                "Number of Columns",
                info['shape'][1]
            )
        with col3:
            st.metric(
                "Memory Usage (MB)",
                f"{info['memory_usage'] / 1024 / 1024:.2f}"
            )
        
        st.markdown("#### Column Types")
        for dtype, count in info['dtypes'].items():
            st.write(f"- {dtype}: {count} columns")
    
    def _display_missing_values(self, missing: Dict):
        """Display missing values analysis"""
        if missing['total_missing'] > 0:
            st.markdown(f"**Total Missing Values:** {missing['total_missing']}")
            
            # Create missing values plot
            fig = go.Figure()
            for col, pct in missing['missing_percentages'].items():
                if pct > 0:
                    fig.add_trace(go.Bar(
                        x=[col],
                        y=[pct],
                        name=col
                    ))
            
            fig.update_layout(
                title="Missing Values by Column (%)",
                xaxis_title="Column",
                yaxis_title="Missing (%)",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values found in the dataset!")
    
    def _display_numerical_analysis(
        self,
        data: pd.DataFrame,
        analysis: Dict
    ):
        """Display numerical features analysis"""
        if analysis:
            # Feature selector
            selected_feature = st.selectbox(
                "Select Feature",
                list(analysis.keys())
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Display statistics
                st.markdown("#### Statistics")
                stats = analysis[selected_feature]
                for stat, value in stats.items():
                    st.write(f"- {stat.title()}: {value:.4f}")
            
            with col2:
                # Display distribution plot
                fig = px.histogram(
                    data,
                    x=selected_feature,
                    title=f"Distribution of {selected_feature}"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _display_categorical_analysis(
        self,
        data: pd.DataFrame,
        analysis: Dict
    ):
        """Display categorical features analysis"""
        if analysis:
            # Feature selector
            selected_feature = st.selectbox(
                "Select Feature",
                list(analysis.keys())
            )
            
            # Display value counts
            fig = px.bar(
                x=list(analysis[selected_feature]['value_counts'].keys()),
                y=list(analysis[selected_feature]['value_counts'].values()),
                title=f"Value Counts for {selected_feature}"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"**Unique Values:** {analysis[selected_feature]['unique_values']}")
    
    def _display_correlations(self, correlations: Dict):
        """Display correlation analysis"""
        if correlations:
            # Correlation heatmap
            fig = px.imshow(
                pd.DataFrame(correlations['correlation_matrix']),
                title="Correlation Heatmap"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # High correlations
            if correlations['high_correlations']:
                st.markdown("#### High Correlations")
                for pair, corr in correlations['high_correlations'].items():
                    st.write(f"- {pair}: {corr:.4f}")
    
    def _display_target_analysis(
        self,
        data: pd.DataFrame,
        target_column: str,
        analysis: Dict
    ):
        """Display target variable analysis"""
        st.markdown(f"**Target Type:** {analysis['type']}")
        
        if analysis['type'] == 'numerical':
            # Distribution plot
            fig = px.histogram(
                data,
                x=target_column,
                title=f"Distribution of {target_column}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.markdown("#### Target Statistics")
            for stat, value in analysis['stats'].items():
                st.write(f"- {stat.title()}: {value:.4f}")
        else:
            # Class distribution plot
            fig = px.pie(
                values=list(analysis['stats']['value_counts'].values()),
                names=list(analysis['stats']['value_counts'].keys()),
                title=f"Class Distribution of {target_column}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Class balance
            st.markdown("#### Class Balance")
            for class_name, balance in analysis['stats']['class_balance'].items():
                st.write(f"- {class_name}: {balance:.2%}")
    
    def _get_high_correlations(
        self,
        corr_matrix: pd.DataFrame,
        threshold: float = 0.8
    ) -> Dict[str, float]:
        """Get highly correlated feature pairs"""
        high_corrs = {}
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr = abs(corr_matrix.iloc[i, j])
                if corr > threshold:
                    pair = f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}"
                    high_corrs[pair] = corr
        
        return high_corrs
