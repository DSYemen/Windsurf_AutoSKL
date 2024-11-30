import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import base64
from PIL import Image
import io
import time
from app.services.data_analyzer import DataAnalyzer
from app.services.model_trainer import ModelTrainer
from app.services.model_evaluator import ModelEvaluator
from app.ui.components.interactive_report import InteractiveReport
from app.ui.components.notifications import NotificationSystem
from app.ui.components.interactive_help import InteractiveHelp
from app.ui.components.model_visualizer import ModelVisualizer
from app.ui.components.model_comparison import ModelComparer, ModelComparison
from app.ui.components.data_analyzer import DataAnalyzer as UIDataAnalyzer

# Initialize components
notification_system = NotificationSystem()
interactive_help = InteractiveHelp()
model_visualizer = ModelVisualizer()
model_comparer = ModelComparer()
interactive_report = InteractiveReport()
data_analyzer = UIDataAnalyzer()

st.set_page_config(
    page_title="AutoSKL Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .plot-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🤖 AutoSKL Dashboard")
    st.sidebar.image("app/ui/assets/logo.png", use_column_width=True)
    
    menu = st.sidebar.selectbox(
        "Navigation",
        ["🏠 Home", "📊 Data Analysis", "🔧 Model Training", "📈 Model Evaluation", "🎯 Predictions"]
    )
    
    if menu == "🏠 Home":
        show_home()
    elif menu == "📊 Data Analysis":
        show_data_analysis()
    elif menu == "🔧 Model Training":
        show_model_training()
    elif menu == "📈 Model Evaluation":
        show_model_evaluation()
    elif menu == "🎯 Predictions":
        show_predictions()

def show_home():
    st.header("Welcome to AutoSKL! 👋")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Models Available", "15+", "All scikit-learn models")
    with col2:
        st.metric("Success Rate", "95%", "Based on user feedback")
    with col3:
        st.metric("Processing Time", "~2 min", "Average training time")
    
    st.markdown("""
    ### 🌟 Key Features
    - **Automated ML Pipeline**: From data to deployment
    - **Smart Model Selection**: Chooses the best model for your data
    - **Advanced Analytics**: Comprehensive data analysis
    - **Interactive Visualizations**: Rich visual insights
    - **Model Explanations**: Understand your models
    
    ### 🚀 Getting Started
    1. Upload your data in the **Data Analysis** section
    2. Review the automated analysis
    3. Train models in the **Model Training** section
    4. Evaluate results in the **Model Evaluation** section
    5. Make predictions with your trained model
    """)

def show_data_analysis():
    st.header("📊 Data Analysis")
    
    # Help button
    if interactive_help.show_help_button("data_analysis"):
        interactive_help.show_help_content("data_analysis")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV, Excel)",
        type=['csv', 'xlsx']
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            # Store in session state
            st.session_state.data = data
            
            # Show success notification
            notification_system.show_success(
                f"Successfully loaded dataset with {data.shape[0]} rows and {data.shape[1]} columns"
            )
            
            # Select target variable
            target_column = st.selectbox(
                "Select Target Variable (Optional)",
                ["None"] + list(data.columns)
            )
            
            # Analyze data
            data_analyzer.show_analysis(
                data,
                target_column if target_column != "None" else None
            )
            
            # Store target column in session state
            if target_column != "None":
                st.session_state.target_column = target_column
            
        except Exception as e:
            notification_system.show_error(
                f"Error loading dataset: {str(e)}"
            )
    else:
        st.info("Please upload a dataset to begin analysis")

def show_model_training():
    st.header("🔧 Model Training")
    
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("Select target column", data.columns)
        with col2:
            task_type = st.selectbox("Select task type", ['auto', 'classification', 'regression'])
        
        model_options = [
            'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm',
            'svc', 'logistic_regression', 'neural_network', 'knn'
        ]
        selected_models = st.multiselect(
            "Select models to try (optional)",
            model_options
        )
        
        if st.button("Start Training"):
            with st.spinner("Training models..."):
                trainer = ModelTrainer()
                X = data.drop(columns=[target_col])
                y = data[target_col]
                
                results = trainer.train(
                    X.values, y.values,
                    model_types=selected_models if selected_models else None
                )
                
                st.success("Training completed!")
                
                # Show results
                st.subheader("📊 Training Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Best Model", results['model_type'])
                with col2:
                    st.metric("Score", f"{results['score']:.4f}")
                
                # Parameters
                st.subheader("⚙️ Best Model Parameters")
                st.json(results['parameters'])
                
                # All Results
                st.subheader("📈 All Models Performance")
                results_df = pd.DataFrame([
                    {
                        'Model': r['model_type'],
                        'Score': r['score'],
                        'Suitability': r['suitability']
                    }
                    for r in results['all_results']
                ])
                
                fig = px.bar(
                    results_df,
                    x='Model',
                    y='Score',
                    color='Suitability',
                    title="Model Performance Comparison"
                )
                st.plotly_chart(fig, use_container_width=True)

def show_model_evaluation():
    st.header("📈 Model Evaluation")
    
    if 'model' not in st.session_state:
        st.warning("Please train a model first in the Model Training section")
        return
    
    model = st.session_state.model
    evaluator = ModelEvaluator()
    
    # Help button
    if interactive_help.show_help_button("model_evaluation"):
        interactive_help.show_help_content("model_evaluation")
    
    # Performance Metrics
    with st.expander("📊 Performance Metrics", expanded=True):
        metrics = evaluator.evaluate_model(
            model,
            st.session_state.X_train,
            st.session_state.X_test,
            st.session_state.y_train,
            st.session_state.y_test
        )
        
        # Display metrics using interactive report
        interactive_report.show_metrics(metrics['metrics'])
        
        # Show notification of evaluation completion
        notification_system.show_success("Model evaluation completed successfully!")
    
    # Model Visualizations
    with st.expander("🎨 Model Visualizations", expanded=True):
        # Use model visualizer for advanced visualizations
        if hasattr(model, "predict_proba"):
            model_visualizer.plot_decision_boundary(
                model,
                st.session_state.X_test,
                st.session_state.y_test
            )
        
        model_visualizer.plot_feature_importance(
            model,
            st.session_state.X_train.columns,
            metrics['feature_importance']
        )
        
        model_visualizer.plot_learning_curves(
            metrics['learning_curves']
        )
        
        # Show confusion matrix and classification report
        if metrics.get('confusion_matrix') is not None:
            interactive_report.show_confusion_matrix(
                metrics['confusion_matrix'],
                metrics.get('class_names')
            )
    
    # Model Comparison
    with st.expander("🔄 Model Comparison", expanded=True):
        if 'trained_models' in st.session_state:
            models_comparison = []
            for model_name, model_info in st.session_state.trained_models.items():
                models_comparison.append(
                    ModelComparison(
                        model_name=model_name,
                        metrics=model_info['metrics'],
                        parameters=model_info['parameters'],
                        training_time=model_info['training_time'],
                        memory_usage=model_info['memory_usage'],
                        feature_importance=model_info.get('feature_importance')
                    )
                )
            
            model_comparer.compare_models(models_comparison)
    
    # Model Explanations
    with st.expander("🔍 Model Explanations", expanded=True):
        if st.session_state.X_test is not None:
            sample_idx = st.slider(
                "Select a sample to explain",
                0, len(st.session_state.X_test)-1
            )
            
            explanations = evaluator.explain_prediction(
                model,
                st.session_state.X_test,
                sample_idx
            )
            
            interactive_report.show_model_explanations(explanations)
    
    # Export Options
    with st.expander("💾 Export Results", expanded=True):
        if st.button("Export Evaluation Results"):
            report_data = {
                'metrics': metrics['metrics'],
                'feature_importance': metrics['feature_importance'],
                'learning_curves': metrics['learning_curves'],
                'visualizations': metrics['visualizations']
            }
            
            interactive_report.export_results(report_data)
            notification_system.show_success("Results exported successfully!")

def show_predictions():
    st.header("🎯 Predictions")
    
    if 'model' not in st.session_state:
        st.warning("Please train a model first in the Model Training section")
        return
    
    # File upload
    uploaded_file = st.file_uploader("Upload data for predictions (CSV)", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        if st.button("Make Predictions"):
            with st.spinner("Making predictions..."):
                predictions = st.session_state.model.predict(data.values)
                
                # Show predictions
                st.subheader("📊 Predictions")
                results_df = pd.DataFrame({
                    'Prediction': predictions
                })
                st.dataframe(results_df)
                
                # Download predictions
                csv = results_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
                st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
