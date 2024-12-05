import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import joblib
from datetime import datetime

st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")

def load_processed_data():
    if "processed_data" not in st.session_state or "target" not in st.session_state:
        st.error("Please complete data preprocessing first!")
        st.stop()
    return st.session_state.processed_data, st.session_state.target

def get_model_class(model_name, problem_type):
    models = {
        'classification': {
            'Random Forest': RandomForestClassifier(),
            'XGBoost': XGBClassifier(),
            'LightGBM': LGBMClassifier(),
            'CatBoost': CatBoostClassifier(verbose=False)
        },
        'regression': {
            'Random Forest': RandomForestRegressor(),
            'XGBoost': XGBRegressor(),
            'LightGBM': LGBMRegressor(),
            'CatBoost': CatBoostRegressor(verbose=False)
        }
    }
    return models[problem_type][model_name]

def evaluate_model(y_true, y_pred, problem_type):
    if problem_type == 'classification':
        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted'),
            'Recall': recall_score(y_true, y_pred, average='weighted'),
            'F1 Score': f1_score(y_true, y_pred, average='weighted')
        }
    else:
        return {
            'R2 Score': r2_score(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))
        }

# Main layout
st.title("🤖 Model Training")

# Load data
df, target = load_processed_data()
X = df.drop(columns=[target])
y = df[target]

# Sidebar
with st.sidebar:
    st.header("Training Configuration")
    
    # Problem type
    problem_type = st.selectbox(
        "Problem Type",
        ["classification", "regression"],
        help="Select the type of machine learning problem"
    )
    
    # Model selection
    model_name = st.selectbox(
        "Select Model",
        ["Random Forest", "XGBoost", "LightGBM", "CatBoost"],
        help="Choose the machine learning algorithm"
    )
    
    # Training parameters
    st.subheader("Training Parameters")
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random State", 0, 999, 42)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataset Overview")
    st.write(f"Features Shape: {X.shape}")
    st.write(f"Target Shape: {y.shape}")
    
    # Feature importance plot
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Train model
            model = get_model_class(model_name, problem_type)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = evaluate_model(y_test, y_pred, problem_type)
            
            # Save model
            model_filename = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            model_path = f"models/{model_filename}"
            joblib.dump(model, model_path)
            
            # Store in session state
            st.session_state.current_model = {
                'model': model,
                'metrics': metrics,
                'feature_importance': pd.Series(
                    model.feature_importances_,
                    index=X.columns
                ).sort_values(ascending=False),
                'filename': model_filename
            }
            
            st.success(f"Model trained and saved as {model_filename}")

with col2:
    if "current_model" in st.session_state:
        # Display metrics
        st.subheader("Model Performance")
        metrics_df = pd.DataFrame(
            st.session_state.current_model['metrics'].items(),
            columns=['Metric', 'Value']
        )
        st.dataframe(metrics_df, hide_index=True)
        
        # Feature importance plot
        st.subheader("Feature Importance")
        fig = px.bar(
            x=st.session_state.current_model['feature_importance'].values,
            y=st.session_state.current_model['feature_importance'].index,
            orientation='h',
            title="Feature Importance"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Download model button
        st.download_button(
            "Download Model",
            data=open(f"models/{st.session_state.current_model['filename']}", 'rb'),
            file_name=st.session_state.current_model['filename'],
            mime="application/octet-stream"
        )
