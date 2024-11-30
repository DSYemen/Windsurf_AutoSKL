import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from pathlib import Path
from datetime import datetime
import time

# Configure page
st.set_page_config(
    page_title="AutoSKL Dashboard",
    page_icon="🤖",
    layout="wide"
)

# API configuration
API_URL = "http://localhost:8000"

def load_css():
    """Load custom CSS"""
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
        }
        .status-box {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .status-success {
            background-color: #dff0d8;
            border: 1px solid #d6e9c6;
            color: #3c763d;
        }
        .status-warning {
            background-color: #fcf8e3;
            border: 1px solid #faebcc;
            color: #8a6d3b;
        }
        .status-error {
            background-color: #f2dede;
            border: 1px solid #ebccd1;
            color: #a94442;
        }
        </style>
    """, unsafe_allow_html=True)

def display_model_status():
    """Display current model status"""
    try:
        response = requests.get(f"{API_URL}/model/status")
        status = response.json()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Model Status",
                "Active" if status["model_available"] else "Not Available"
            )
            
        with col2:
            st.metric(
                "Monitoring",
                "Enabled" if status["monitoring_enabled"] else "Disabled"
            )
            
        if "update_status" in status:
            with col3:
                update_status = status["update_status"]
                st.metric(
                    "Model Version",
                    f"v{update_status['current_version']}"
                )
                
            if update_status["performance_history"]:
                # Plot performance history
                df = pd.DataFrame(update_status["performance_history"])
                fig = px.line(
                    df,
                    x="timestamp",
                    y="score",
                    title="Model Performance History"
                )
                st.plotly_chart(fig, use_container_width=True)
                
    except Exception as e:
        st.error(f"Error fetching model status: {str(e)}")

def upload_and_train():
    """Handle model training"""
    st.header("Train New Model")
    
    uploaded_file = st.file_uploader(
        "Upload training data (CSV)",
        type=["csv"]
    )
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_column = st.selectbox(
                "Select target column",
                df.columns.tolist()
            )
            
        with col2:
            n_trials = st.slider(
                "Number of optimization trials",
                min_value=10,
                max_value=500,
                value=100
            )
            
        if st.button("Train Model"):
            try:
                with st.spinner("Training model..."):
                    files = {"file": uploaded_file}
                    data = {
                        "target_column": target_column,
                        "n_trials": n_trials
                    }
                    
                    response = requests.post(
                        f"{API_URL}/train",
                        files=files,
                        data=data
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("Model trained successfully!")
                        
                        # Display training results
                        st.subheader("Training Results")
                        st.json(result["training_results"])
                        
                        # Display report if available
                        if "report_path" in result:
                            with open(result["report_path"], "r") as f:
                                st.components.v1.html(
                                    f.read(),
                                    height=800,
                                    scrolling=True
                                )
                    else:
                        st.error(f"Training failed: {response.text}")
                        
            except Exception as e:
                st.error(f"Error during training: {str(e)}")

def make_predictions():
    """Handle predictions"""
    st.header("Make Predictions")
    
    uploaded_file = st.file_uploader(
        "Upload data for predictions (CSV)",
        type=["csv"],
        key="prediction_upload"
    )
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        
        if st.button("Make Predictions"):
            try:
                with st.spinner("Making predictions..."):
                    data = {"data": df.to_dict(orient="records")}
                    response = requests.post(
                        f"{API_URL}/predict",
                        json=data
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display predictions
                        predictions = pd.DataFrame({
                            "Prediction": result["predictions"]
                        })
                        st.success("Predictions made successfully!")
                        st.dataframe(predictions)
                        
                        # Display monitoring information
                        if "monitoring" in result:
                            st.subheader("Monitoring Information")
                            
                            # Display drift analysis
                            if "drift_analysis" in result["monitoring"]:
                                drift = result["monitoring"]["drift_analysis"]
                                if drift["drift_detected"]:
                                    st.warning("Data drift detected!")
                                    
                                # Plot drift scores
                                drift_df = pd.DataFrame(
                                    drift["feature_drift_scores"]
                                )
                                fig = px.bar(
                                    drift_df,
                                    x="feature_index",
                                    y="statistic",
                                    title="Feature Drift Analysis"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                    else:
                        st.error(f"Prediction failed: {response.text}")
                        
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

def view_reports():
    """Display available reports"""
    st.header("Model Reports")
    
    try:
        response = requests.get(f"{API_URL}/reports")
        if response.status_code == 200:
            reports = response.json()["reports"]
            
            if not reports:
                st.info("No reports available")
                return
                
            for report in reports:
                with st.expander(f"{report['name']} ({report['created']})"):
                    try:
                        with open(report["path"], "r") as f:
                            st.components.v1.html(
                                f.read(),
                                height=600,
                                scrolling=True
                            )
                    except Exception as e:
                        st.error(f"Error loading report: {str(e)}")
                        
    except Exception as e:
        st.error(f"Error fetching reports: {str(e)}")

def main():
    """Main dashboard application"""
    load_css()
    
    st.title("AutoSKL Dashboard 🤖")
    st.markdown("""
        Welcome to the AutoSKL Dashboard! This interface allows you to:
        - Train new machine learning models
        - Make predictions using trained models
        - Monitor model performance and data drift
        - View detailed reports
    """)
    
    # Display model status
    display_model_status()
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Train Model", "Make Predictions", "View Reports"]
    )
    
    if page == "Train Model":
        upload_and_train()
    elif page == "Make Predictions":
        make_predictions()
    else:
        view_reports()

if __name__ == "__main__":
    main()
