import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import cross_val_score
import shap
from datetime import datetime

st.set_page_config(page_title="Model Evaluation", page_icon="📊", layout="wide")

def load_model_data():
    if "current_model" not in st.session_state:
        st.error("Please train a model first!")
        st.stop()
    return st.session_state.current_model

def plot_confusion_matrix(y_true, y_pred, classes):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create labels for the confusion matrix
    labels = [str(c) for c in classes]
    
    # Create figure
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        showscale=True,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverongaps=False
    ))
    
    # Update layout
    fig.update_layout(
        title='Confusion Matrix',
        xaxis=dict(
            title='Predicted',
            side='bottom'
        ),
        yaxis=dict(
            title='Actual',
            autorange='reversed'  # This ensures the matrix is shown in the correct orientation
        ),
        width=600,
        height=600
    )
    
    return fig

def plot_roc_curve(y_true, y_prob):
    if y_prob.shape[1] == 2:  # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        auc_score = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr,
                name=f'ROC curve (AUC = {auc_score:.2f})',
                mode='lines'
            )
        )
        
    else:  # Multiclass classification
        fig = go.Figure()
        for i in range(y_prob.shape[1]):
            fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
            auc_score = auc(fpr, tpr)
            
            fig.add_trace(
                go.Scatter(
                    x=fpr, y=tpr,
                    name=f'Class {i} (AUC = {auc_score:.2f})',
                    mode='lines'
                )
            )
    
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random',
            mode='lines',
            line=dict(dash='dash', color='gray')
        )
    )
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True
    )
    return fig

# Main layout
st.title("📊 Model Evaluation")

# Load model
model_data = load_model_data()
if model_data:
    model = model_data['model']
    metrics = model_data.get('metrics', {})
    
    # Get data
    X = st.session_state.processed_data.drop(columns=[st.session_state.target])
    y = st.session_state.processed_data[st.session_state.target]
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Determine problem type
    is_classifier = hasattr(model, 'classes_')
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        if is_classifier:
            st.subheader("Classification Metrics")
            
            # Display metrics
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                'Value': [
                    accuracy_score(y, y_pred),
                    precision_score(y, y_pred, average='weighted'),
                    recall_score(y, y_pred, average='weighted'),
                    f1_score(y, y_pred, average='weighted')
                ]
            })
            st.dataframe(metrics_df, use_container_width=True)
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            st.plotly_chart(
                plot_confusion_matrix(y, y_pred, model.classes_),
                use_container_width=True
            )
            
            # ROC Curve for classifiers with predict_proba
            if hasattr(model, 'predict_proba'):
                st.subheader("ROC Curve")
                y_prob = model.predict_proba(X)
                st.plotly_chart(plot_roc_curve(y, y_prob), use_container_width=True)
        
        else:
            st.subheader("Regression Metrics")
            
            # Display metrics
            metrics_df = pd.DataFrame({
                'Metric': ['R² Score', 'MAE', 'MSE', 'RMSE'],
                'Value': [
                    r2_score(y, y_pred),
                    mean_absolute_error(y, y_pred),
                    mean_squared_error(y, y_pred),
                    mean_squared_error(y, y_pred, squared=False)
                ]
            })
            st.dataframe(metrics_df, use_container_width=True)
            
            # Actual vs Predicted Plot
            st.subheader("Actual vs Predicted")
            fig = px.scatter(
                x=y, y=y_pred,
                labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                title='Actual vs Predicted Values'
            )
            fig.add_trace(
                go.Scatter(
                    x=[y.min(), y.max()],
                    y=[y.min(), y.max()],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='gray')
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Residuals Plot
            st.subheader("Residuals Plot")
            residuals = y - y_pred
            fig = px.scatter(
                x=y_pred, y=residuals,
                labels={'x': 'Predicted Values', 'y': 'Residuals'},
                title='Residuals Plot'
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # SHAP Values
        if st.checkbox("Show SHAP Values"):
            st.subheader("SHAP Values")
            try:
                # Create SHAP explainer
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                
                # Handle different types of SHAP values
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if is_classifier else shap_values
                
                # Create SHAP summary plot
                plt.figure(figsize=(10, 6))
                shap.summary_plot(
                    shap_values,
                    X,
                    plot_type="bar",
                    show=False
                )
                fig1 = plt.gcf()
                st.pyplot(fig1)
                plt.close(fig1)
                
                # Create SHAP beeswarm plot
                plt.figure(figsize=(10, 6))
                shap.summary_plot(
                    shap_values,
                    X,
                    plot_type="dot",
                    show=False
                )
                fig2 = plt.gcf()
                st.pyplot(fig2)
                plt.close(fig2)
                
                # Add SHAP force plot for first prediction
                if st.checkbox("Show SHAP Force Plot"):
                    st.subheader("SHAP Force Plot (First Prediction)")
                    plt.figure(figsize=(10, 3))
                    shap.force_plot(
                        explainer.expected_value[1] if is_classifier else explainer.expected_value,
                        shap_values[0],
                        X.iloc[0],
                        matplotlib=True,
                        show=False
                    )
                    fig3 = plt.gcf()
                    st.pyplot(fig3)
                    plt.close(fig3)
                    
            except Exception as e:
                st.error(f"Error calculating SHAP values: {str(e)}")
        
        # Model Parameters
        st.subheader("Model Parameters")
        st.json(model.get_params())
        
        # Download Model Report
        st.subheader("Download Report")
        if st.button("Generate Report"):
            report = {
                'model_type': type(model).__name__,
                'parameters': model.get_params(),
                'metrics': metrics,
                'feature_importance': importance_df.to_dict() if hasattr(model, 'feature_importances_') else None,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            st.download_button(
                "Download Model Report",
                json.dumps(report, indent=2),
                file_name="model_report.json",
                mime="application/json"
            )

# Sidebar
with st.sidebar:
    st.header("Evaluation Options")
    
    eval_type = st.radio(
        "Evaluation Type",
        ["Basic Metrics", "Cross Validation", "Feature Analysis"]
    )

# Main content
if eval_type == "Basic Metrics":
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance Metrics")
        metrics_df = pd.DataFrame(
            model_data['metrics'].items(),
            columns=['Metric', 'Value']
        )
        st.dataframe(metrics_df, hide_index=True)
        
        if hasattr(model, 'predict_proba'):
            st.subheader("ROC Curve")
            y_prob = model.predict_proba(X)
            st.plotly_chart(plot_roc_curve(y, y_prob), use_container_width=True)
    
    with col2:
        if hasattr(model, 'classes_'):
            st.subheader("Confusion Matrix")
            y_pred = model.predict(X)
            st.plotly_chart(
                plot_confusion_matrix(y, y_pred, model.classes_),
                use_container_width=True
            )

elif eval_type == "Cross Validation":
    st.subheader("Cross Validation Results")
    
    n_folds = st.slider("Number of Folds", 3, 10, 5)
    
    with st.spinner("Performing cross validation..."):
        cv_scores = cross_val_score(model, X, y, cv=n_folds)
        
        st.metric("Mean CV Score", f"{cv_scores.mean():.3f}")
        st.metric("CV Score Std", f"{cv_scores.std():.3f}")
        
        fig = px.box(
            cv_scores,
            title="Cross Validation Scores Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

else:  # Feature Analysis
    st.subheader("Feature Importance Analysis")
    
    # Standard feature importance
    st.plotly_chart(
        px.bar(
            x=model_data['feature_importance'].values,
            y=model_data['feature_importance'].index,
            orientation='h',
            title="Feature Importance"
        ),
        use_container_width=True
    )
    
    # SHAP values
    if st.button("Calculate SHAP Values"):
        with st.spinner("Calculating SHAP values..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            st.pyplot(shap.summary_plot(shap_values, X, plot_type="bar", show=False))
            st.pyplot(shap.summary_plot(shap_values, X, show=False))
