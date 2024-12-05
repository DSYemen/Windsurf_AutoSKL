import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import optuna
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import joblib
from datetime import datetime
import pathlib

st.set_page_config(page_title="AutoML", page_icon="⚡", layout="wide")

def load_processed_data():
    if "processed_data" not in st.session_state or "target" not in st.session_state:
        st.error("Please complete data preprocessing first!")
        st.stop()
    return st.session_state.processed_data, st.session_state.target

def create_model(trial, model_name, problem_type):
    if model_name == "Random Forest":
        if problem_type == "classification":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
            }
            return RandomForestClassifier(**params)
        else:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
            }
            return RandomForestRegressor(**params)
    
    elif model_name == "XGBoost":
        if problem_type == "classification":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0)
            }
            return XGBClassifier(**params)
        else:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0)
            }
            return XGBRegressor(**params)
    
    elif model_name == "LightGBM":
        if problem_type == "classification":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100)
            }
            return LGBMClassifier(**params)
        else:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100)
            }
            return LGBMRegressor(**params)
    
    else:  # CatBoost
        if problem_type == "classification":
            params = {
                'iterations': trial.suggest_int('iterations', 50, 300),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True)
            }
            return CatBoostClassifier(**params, verbose=False)
        else:
            params = {
                'iterations': trial.suggest_int('iterations', 50, 300),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True)
            }
            return CatBoostRegressor(**params, verbose=False)

def objective(trial, X_train, X_test, y_train, y_test, model_name, problem_type):
    model = create_model(trial, model_name, problem_type)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if problem_type == "classification":
        return accuracy_score(y_test, y_pred)
    else:
        return r2_score(y_test, y_pred)

# Main layout
st.title("⚡ AutoML - Automated Model Optimization")

# Load data
df, target = load_processed_data()
X = df.drop(columns=[target])
y = df[target]

# Sidebar
with st.sidebar:
    st.header("AutoML Configuration")
    
    problem_type = st.selectbox(
        "Problem Type",
        ["classification", "regression"]
    )
    
    models_to_try = st.multiselect(
        "Select Models to Try",
        ["Random Forest", "XGBoost", "LightGBM", "CatBoost"],
        default=["Random Forest", "XGBoost"]
    )
    
    n_trials = st.slider(
        "Number of Trials per Model",
        min_value=10,
        max_value=100,
        value=30
    )
    
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)

# Main content
if st.button("Start AutoML Optimization"):
    if not models_to_try:
        st.error("Please select at least one model to try!")
    else:
        # Create models directory if it doesn't exist
        models_dir = pathlib.Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Store results
        results = []
        best_model_info = None
        best_score = float('-inf')
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Try each model
        for i, model_name in enumerate(models_to_try):
            status_text.text(f"Optimizing {model_name}...")
            
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: objective(
                    trial, X_train, X_test, y_train, y_test,
                    model_name, problem_type
                ),
                n_trials=n_trials
            )
            
            # Create best model
            best_model = create_model(
                study.best_trial, model_name, problem_type
            )
            best_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = best_model.predict(X_test)
            if problem_type == "classification":
                score = accuracy_score(y_test, y_pred)
                metric_name = "Accuracy"
            else:
                score = r2_score(y_test, y_pred)
                metric_name = "R2 Score"
            
            # Save model info
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"automl_{model_name.lower().replace(' ', '_')}_{timestamp}.joblib"
            model_path = models_dir / model_filename
            
            # Save model
            joblib.dump(best_model, model_path)
            
            model_info = {
                'Model': model_name,
                metric_name: score,
                'Best Parameters': study.best_params,
                'model': best_model,
                'metrics': {metric_name: score},
                'feature_importance': pd.Series(
                    best_model.feature_importances_,
                    index=X.columns
                ).sort_values(ascending=False) if hasattr(best_model, 'feature_importances_') else None,
                'filename': model_filename
            }
            
            results.append(model_info)
            
            # Update best model if score is better
            if score > best_score:
                best_score = score
                best_model_info = model_info
            
            # Update progress
            progress_bar.progress((i + 1) / len(models_to_try))
        
        # Show results
        st.success("AutoML optimization completed!")
        
        # Results table
        st.subheader("Model Comparison")
        results_df = pd.DataFrame([
            {'Model': r['Model'], metric_name: r[metric_name]}
            for r in results
        ])
        st.dataframe(results_df, use_container_width=True)
        
        # Plot results
        fig = px.bar(
            results_df,
            x='Model',
            y=metric_name,
            title=f"Model Performance Comparison ({metric_name})"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model details
        if best_model_info:
            st.subheader(f"Best Model: {best_model_info['Model']}")
            st.json(best_model_info['Best Parameters'])
            
            # Save best model to session state
            st.session_state.current_model = {
                'model': best_model_info['model'],
                'metrics': best_model_info['metrics'],
                'feature_importance': best_model_info['feature_importance'],
                'filename': best_model_info['filename']
            }
            
            st.success(f"Best model saved as {best_model_info['filename']}. You can now proceed to Model Evaluation!")
