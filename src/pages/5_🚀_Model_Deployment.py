import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import base64
from datetime import datetime

st.set_page_config(page_title="Model Deployment", page_icon="🚀", layout="wide")

def load_model():
    if "current_model" not in st.session_state:
        st.warning("No model currently trained. Please train a model first!")
        return None
    return st.session_state.current_model

def create_prediction_api(model, feature_names):
    api_code = f'''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import List, Dict

app = FastAPI(title="Model Prediction API")

class PredictionInput(BaseModel):
    features: Dict[str, float]

class PredictionOutput(BaseModel):
    prediction: float
    probability: List[float] = None

# Load the model
model = joblib.load("{model['filename']}")

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([input_data.features])
        
        # Ensure correct feature order
        feature_names = {feature_names}
        df = df.reindex(columns=feature_names, fill_value=0)
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        # Get probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(df)[0].tolist()
        
        return PredictionOutput(
            prediction=float(prediction),
            probability=probability
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
'''
    return api_code

def create_docker_file():
    return '''
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
COPY model.joblib .
COPY api.py .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
'''

def create_requirements_file(model):
    return f'''
fastapi>=0.110.0
uvicorn>=0.27.0
pandas>=2.2.0
scikit-learn>={model.__module__.split('.')[1]}
joblib>=1.3.2
python-multipart>=0.0.9
'''

# Main layout
st.title("🚀 Model Deployment")

# Load model
model_data = load_model()

if model_data:
    # Sidebar
    with st.sidebar:
        st.header("Deployment Options")
        deployment_type = st.radio(
            "Deployment Type",
            ["FastAPI", "Batch Predictions", "Export Model"]
        )
    
    # Main content
    if deployment_type == "FastAPI":
        st.subheader("FastAPI Deployment")
        
        # Generate API code
        api_code = create_prediction_api(
            model_data,
            list(st.session_state.processed_data.drop(columns=[st.session_state.target]).columns)
        )
        
        # Display API code
        st.code(api_code, language="python")
        
        # Generate Dockerfile
        dockerfile = create_docker_file()
        requirements = create_requirements_file(model_data['model'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dockerfile")
            st.code(dockerfile, language="dockerfile")
        
        with col2:
            st.subheader("Requirements")
            st.code(requirements, language="text")
        
        # Download deployment package
        if st.button("Download Deployment Package"):
            # Create deployment directory
            deploy_dir = Path("deployment")
            deploy_dir.mkdir(exist_ok=True)
            
            # Save files
            with open(deploy_dir / "api.py", "w") as f:
                f.write(api_code)
            with open(deploy_dir / "Dockerfile", "w") as f:
                f.write(dockerfile)
            with open(deploy_dir / "requirements.txt", "w") as f:
                f.write(requirements)
            
            # Copy model file
            model_path = Path("models") / model_data['filename']
            if model_path.exists():
                joblib.dump(model_data['model'], deploy_dir / "model.joblib")
            
            st.success("Deployment package created successfully!")
            
    elif deployment_type == "Batch Predictions":
        st.subheader("Batch Predictions")
        
        uploaded_file = st.file_uploader("Upload prediction data", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            # Load prediction data
            try:
                if uploaded_file.name.endswith('.csv'):
                    pred_data = pd.read_csv(uploaded_file)
                else:
                    pred_data = pd.read_excel(uploaded_file)
                
                # Make predictions
                predictions = model_data['model'].predict(pred_data)
                
                # Add predictions to dataframe
                pred_data['Predictions'] = predictions
                
                # Display results
                st.dataframe(pred_data, use_container_width=True)
                
                # Download predictions
                csv = pred_data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error making predictions: {str(e)}")
    
    else:  # Export Model
        st.subheader("Export Model")
        
        # Model info
        st.json({
            'model_type': type(model_data['model']).__name__,
            'training_date': model_data['filename'].split('_')[1].split('.')[0],
            'metrics': model_data['metrics']
        })
        
        # Download model
        if st.button("Download Model"):
            model_path = Path("models") / model_data['filename']
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    model_bytes = f.read()
                st.download_button(
                    "Download Model File",
                    model_bytes,
                    model_data['filename'],
                    "application/octet-stream"
                )
            else:
                st.error("Model file not found!")
