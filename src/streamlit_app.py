import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd

# Configure the Streamlit page
st.set_page_config(
    page_title="التعلم الآلي التلقائي",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
with open('static/css/rtl.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Add Cairo font
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@200;300;400;600;700;900&display=swap" rel="stylesheet">
    <style>
        * {font-family: 'Cairo', sans-serif !important;}
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🌊 Windsurf AutoSKL")
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["Dashboard", "Models", "Experiments", "Settings"]
    )
    
    st.markdown("---")
    
    # Time Range Selector
    time_range = st.selectbox(
        "Time Range",
        ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Custom Range"]
    )
    
    if time_range == "Custom Range":
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
        end_date = st.date_input("End Date", datetime.now())

# Mock data (replace with real data later)
mock_stats = {
    "total_models": 42,
    "model_growth": 15,
    "active_experiments": 8,
    "total_predictions": 12500,
    "prediction_growth": 25,
    "system_health": 98
}

# Mock performance data
model_performance = pd.DataFrame({
    "Model": ["Model A", "Model B", "Model C", "Model D"],
    "Accuracy": [0.95, 0.87, 0.92, 0.89],
    "F1 Score": [0.94, 0.86, 0.91, 0.88]
})

# Mock prediction volume data
prediction_volume = pd.DataFrame({
    "Date": pd.date_range(start=datetime.now() - timedelta(days=6), end=datetime.now(), freq='D'),
    "Predictions": [1200, 1900, 1500, 1800, 2100, 1700, 1600]
})

# Main content
if page == "Dashboard":
    # Header
    st.title("Dashboard")
    st.markdown("---")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Models",
            mock_stats["total_models"],
            f"{mock_stats['model_growth']}%"
        )
    
    with col2:
        st.metric(
            "Active Experiments",
            mock_stats["active_experiments"],
            None
        )
    
    with col3:
        st.metric(
            "Total Predictions",
            f"{mock_stats['total_predictions']:,}",
            f"{mock_stats['prediction_growth']}%"
        )
    
    with col4:
        st.metric(
            "System Health",
            f"{mock_stats['system_health']}%",
            "Healthy" if mock_stats['system_health'] >= 90 else "Warning"
        )
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance")
        fig = px.bar(
            model_performance,
            x="Model",
            y=["Accuracy", "F1 Score"],
            barmode="group",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Prediction Volume")
        fig = px.line(
            prediction_volume,
            x="Date",
            y="Predictions",
            template="plotly_white"
        )
        fig.update_traces(fill='tozeroy')
        st.plotly_chart(fig, use_container_width=True)

elif page == "Models":
    st.title("Models")
    # Add models page content here

elif page == "Experiments":
    st.title("Experiments")
    # Add experiments page content here

elif page == "Settings":
    st.title("Settings")
    # Add settings page content here
