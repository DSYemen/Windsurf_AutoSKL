from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import json
from pathlib import Path
import os
from datetime import datetime
from app.core.config import settings
from app.services.data_processor import DataProcessor
from app.services.model_trainer import ModelTrainer
from app.services.model_monitor import ModelMonitor
from app.services.report_generator import ReportGenerator
from app.services.auto_updater import AutoUpdater

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
data_processor = DataProcessor()
model_trainer = ModelTrainer()
model_monitor = ModelMonitor()
report_generator = ReportGenerator()
auto_updater = AutoUpdater()

class TrainingRequest(BaseModel):
    target_column: str
    model_types: Optional[List[str]] = None
    n_trials: Optional[int] = 100

class PredictionRequest(BaseModel):
    data: List[Dict[str, Any]]

@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    request: TrainingRequest = None
):
    """Train a model using uploaded data"""
    try:
        # Read data
        df = pd.read_csv(file.file)
        
        # Process data
        X, y = data_processor.fit_transform(df, request.target_column)
        
        # Train model
        training_results = model_trainer.train(
            X, y,
            model_types=request.model_types,
            n_trials=request.n_trials
        )
        
        # Save model
        os.makedirs(settings.MODEL_PATH, exist_ok=True)
        model_path = Path(settings.MODEL_PATH) / "model.joblib"
        model_trainer.save_model(str(model_path))
        
        # Set reference data for monitoring
        model_monitor.set_reference_data(X)
        
        # Generate training report
        feature_importance = {f"feature_{i}": imp for i, imp in enumerate(model_trainer.best_model.feature_importances_)}
        performance_metrics = {"score": training_results["score"]}
        validation_results = {}
        
        report_path = report_generator.generate_training_report(
            training_results,
            feature_importance,
            performance_metrics,
            validation_results
        )
        
        return {
            "status": "success",
            "training_results": training_results,
            "report_path": report_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Make predictions using trained model"""
    try:
        # Load model if not already loaded
        if model_trainer.best_model is None:
            model_path = Path(settings.MODEL_PATH) / "model.joblib"
            if not model_path.exists():
                raise HTTPException(
                    status_code=400,
                    detail="No trained model found. Please train a model first."
                )
            model_trainer = ModelTrainer.load_model(str(model_path))
        
        # Transform input data
        df = pd.DataFrame(request.data)
        X = data_processor.transform(df)
        
        # Make predictions
        predictions = model_trainer.predict(X)
        
        # Monitor predictions
        monitoring_data = {
            "predictions": predictions.tolist(),
            "timestamp": datetime.now().isoformat()
        }
        
        if settings.ENABLE_MONITORING:
            # Detect drift
            drift_report = model_monitor.detect_drift(X)
            monitoring_data["drift_analysis"] = drift_report
            
            # Check if model update is needed
            monitoring_report = auto_updater.collect_new_data(X)
            if auto_updater.should_update(monitoring_report):
                # Trigger async model update
                auto_updater.update_model.delay(X, predictions)
        
        return {
            "status": "success",
            "predictions": predictions.tolist(),
            "monitoring": monitoring_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/status")
async def get_model_status():
    """Get current model status and monitoring information"""
    try:
        model_path = Path(settings.MODEL_PATH) / "model.joblib"
        
        status = {
            "model_available": model_path.exists(),
            "monitoring_enabled": settings.ENABLE_MONITORING
        }
        
        # Get auto-update status
        if model_path.exists():
            status["update_status"] = auto_updater.get_update_status()
        
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reports")
async def get_reports():
    """Get list of available reports"""
    try:
        reports_dir = Path("reports")
        if not reports_dir.exists():
            return {"reports": []}
            
        reports = []
        for report_file in reports_dir.glob("*.html"):
            reports.append({
                "name": report_file.stem,
                "path": str(report_file),
                "type": "html",
                "created": datetime.fromtimestamp(report_file.stat().st_mtime).isoformat()
            })
            
        return {"reports": reports}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
