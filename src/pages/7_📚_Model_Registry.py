import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import plotly.express as px
from datetime import datetime
import shutil

st.set_page_config(page_title="Model Registry", page_icon="📚", layout="wide")

def load_model_metadata(model_path):
    try:
        model = joblib.load(model_path)
        creation_time = datetime.fromtimestamp(model_path.stat().st_ctime)
        
        metadata = {
            'name': model_path.stem,
            'type': type(model).__name__,
            'created': creation_time.strftime('%Y-%m-%d %H:%M:%S'),
            'size': f"{model_path.stat().st_size / (1024*1024):.2f} MB",
            'path': str(model_path)
        }
        
        if hasattr(model, 'feature_importances_'):
            metadata['has_feature_importance'] = True
        
        return metadata
    except Exception as e:
        st.error(f"Error loading model {model_path}: {str(e)}")
        return None

def compare_models(models_data):
    comparison_data = []
    for model_data in models_data:
        if 'metrics' in model_data:
            metrics = model_data['metrics']
            comparison_data.append({
                'Model': model_data['name'],
                **metrics
            })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        
        # Plot comparison
        for metric in df.columns[1:]:  # Skip 'Model' column
            fig = px.bar(
                df,
                x='Model',
                y=metric,
                title=f"Model Comparison - {metric}"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No metrics available for comparison")

# Main layout
st.title("📚 Model Registry")

# Sidebar
with st.sidebar:
    st.header("Registry Options")
    
    view_mode = st.radio(
        "View Mode",
        ["List View", "Comparison View", "Model Details"]
    )
    
    sort_by = st.selectbox(
        "Sort By",
        ["Created Date", "Name", "Type", "Size"]
    )

# Load models
models_dir = Path("models")
if not models_dir.exists():
    st.error("Models directory not found!")
    st.stop()

model_files = list(models_dir.glob("*.joblib"))
if not model_files:
    st.warning("No models found in registry!")
    st.stop()

# Load model metadata
models_metadata = []
for model_path in model_files:
    metadata = load_model_metadata(model_path)
    if metadata:
        models_metadata.append(metadata)

# Sort models
if sort_by == "Created Date":
    models_metadata.sort(key=lambda x: x['created'], reverse=True)
elif sort_by == "Name":
    models_metadata.sort(key=lambda x: x['name'])
elif sort_by == "Type":
    models_metadata.sort(key=lambda x: x['type'])
else:  # Size
    models_metadata.sort(key=lambda x: float(x['size'].split()[0]), reverse=True)

# Main content
if view_mode == "List View":
    # Create DataFrame for display
    df = pd.DataFrame(models_metadata)
    st.dataframe(df, use_container_width=True)
    
    # Actions for selected models
    selected_models = st.multiselect(
        "Select Models for Action",
        df['name'].tolist()
    )
    
    if selected_models:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Download Selected"):
                for model_name in selected_models:
                    model_path = models_dir / f"{model_name}.joblib"
                    if model_path.exists():
                        with open(model_path, 'rb') as f:
                            st.download_button(
                                f"Download {model_name}",
                                f,
                                file_name=f"{model_name}.joblib",
                                mime="application/octet-stream"
                            )
        
        with col2:
            if st.button("Delete Selected"):
                for model_name in selected_models:
                    model_path = models_dir / f"{model_name}.joblib"
                    if model_path.exists():
                        model_path.unlink()
                st.success("Selected models deleted!")
                st.experimental_rerun()
        
        with col3:
            if st.button("Export Metadata"):
                metadata_dict = {
                    model['name']: model for model in models_metadata
                    if model['name'] in selected_models
                }
                st.download_button(
                    "Download Metadata JSON",
                    json.dumps(metadata_dict, indent=2),
                    file_name="model_metadata.json",
                    mime="application/json"
                )

elif view_mode == "Comparison View":
    if len(models_metadata) < 2:
        st.warning("Need at least 2 models for comparison!")
    else:
        compare_models(models_metadata)

else:  # Model Details
    # Select model
    selected_model = st.selectbox(
        "Select Model",
        [m['name'] for m in models_metadata]
    )
    
    if selected_model:
        model_data = next(m for m in models_metadata if m['name'] == selected_model)
        
        # Display model details
        st.subheader("Model Information")
        st.json(model_data)
        
        # Load model for additional information
        model_path = Path(model_data['path'])
        if model_path.exists():
            model = joblib.load(model_path)
            
            # Feature importance if available
            if model_data.get('has_feature_importance'):
                st.subheader("Feature Importance")
                feature_importance = pd.Series(
                    model.feature_importances_,
                    index=model.feature_names_in_
                ).sort_values(ascending=False)
                
                fig = px.bar(
                    x=feature_importance.values,
                    y=feature_importance.index,
                    orientation='h',
                    title="Feature Importance"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Model parameters
            st.subheader("Model Parameters")
            st.json(model.get_params())
            
            # Export options
            st.subheader("Export Options")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Download Model"):
                    with open(model_path, 'rb') as f:
                        st.download_button(
                            "Download Model File",
                            f,
                            file_name=f"{selected_model}.joblib",
                            mime="application/octet-stream"
                        )
            
            with col2:
                if st.button("Export Deployment Package"):
                    # Create deployment directory
                    deploy_dir = Path("deployment") / selected_model
                    deploy_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy model file
                    shutil.copy2(model_path, deploy_dir / "model.joblib")
                    
                    # Create requirements.txt
                    with open(deploy_dir / "requirements.txt", "w") as f:
                        f.write(f"""
scikit-learn=={model.__module__.split('.')[1]}
pandas>=2.2.0
numpy>=1.26.0
joblib>=1.3.2
                        """.strip())
                    
                    # Create README
                    with open(deploy_dir / "README.md", "w") as f:
                        f.write(f"""
# {selected_model} Deployment Package

## Model Information
- Type: {model_data['type']}
- Created: {model_data['created']}
- Size: {model_data['size']}

## Usage
1. Install requirements: `pip install -r requirements.txt`
2. Load model:
```python
import joblib
model = joblib.load('model.joblib')
```
                        """.strip())
                    
                    # Zip the deployment package
                    shutil.make_archive(str(deploy_dir), 'zip', deploy_dir)
                    
                    # Offer download
                    with open(str(deploy_dir) + '.zip', 'rb') as f:
                        st.download_button(
                            "Download Deployment Package",
                            f,
                            file_name=f"{selected_model}_deployment.zip",
                            mime="application/zip"
                        )
                    
                    st.success("Deployment package created successfully!")
