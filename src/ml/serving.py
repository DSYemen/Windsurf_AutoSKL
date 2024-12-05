from typing import Dict, Any, Optional, List
import json
from pathlib import Path
import joblib
import pandas as pd
from fastapi import HTTPException
from sqlalchemy.orm import Session
from src.ml.base import BaseMLModel
from src.ml.classification import ClassificationModel
from src.ml.regression import RegressionModel
from src.ml.clustering import ClusteringModel
from src.ml.dimensionality_reduction import DimensionalityReductionModel
from src.ml.registry import ModelRegistry
from src.models.ml_model import MLModel

class ModelServer:
    def __init__(self, db_session: Session):
        self.db = db_session
        self.registry = ModelRegistry(db_session)
        self.loaded_models: Dict[str, BaseMLModel] = {}
        
    def _load_model(self, model_id: int, version: Optional[str] = None) -> BaseMLModel:
        """Load model from registry"""
        # Check if model is already loaded
        model_key = f"{model_id}_{version if version else 'current'}"
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
            
        # Get model metadata
        db_model = self.db.query(MLModel).filter_by(id=model_id).first()
        if not db_model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            
        # Get model version
        model_version = self.registry.get_model_version(model_id, version)
        if not model_version:
            raise HTTPException(status_code=404, detail="Model version not found")
            
        # Get model files
        model_files = self.registry.get_model_files(model_id, model_version.version)
        if not model_files:
            raise HTTPException(status_code=404, detail="Model files not found")
            
        # Load model based on type
        model_class = {
            'classification': ClassificationModel,
            'regression': RegressionModel,
            'clustering': ClusteringModel,
            'dimensionality_reduction': DimensionalityReductionModel
        }.get(db_model.type)
        
        if not model_class:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {db_model.type}")
            
        # Create model instance
        model = model_class(
            model_name=db_model.name,
            user_id=db_model.user_id,
            algorithm=db_model.algorithm,
            params=model_version.parameters
        )
        
        # Load model state
        model_data = joblib.load(model_files['model.joblib'])
        model.model = model_data['model']
        model.metadata = model_data['metadata']
        
        # Cache loaded model
        self.loaded_models[model_key] = model
        return model
        
    async def predict(
        self,
        model_id: int,
        data: Dict[str, Any],
        version: Optional[str] = None,
        return_proba: bool = False
    ) -> Dict[str, Any]:
        """Make predictions using model"""
        # Load model
        model = self._load_model(model_id, version)
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(input_df)
        result = {'prediction': prediction.tolist()}
        
        # Add probability scores for classification
        if return_proba and isinstance(model, ClassificationModel):
            probabilities = model.predict_proba(input_df)
            result['probabilities'] = probabilities.tolist()
            
        return result
        
    async def batch_predict(
        self,
        model_id: int,
        data: List[Dict[str, Any]],
        version: Optional[str] = None,
        return_proba: bool = False
    ) -> Dict[str, Any]:
        """Make batch predictions"""
        # Load model
        model = self._load_model(model_id, version)
        
        # Convert input to DataFrame
        input_df = pd.DataFrame(data)
        
        # Make predictions
        predictions = model.predict(input_df)
        result = {'predictions': predictions.tolist()}
        
        # Add probability scores for classification
        if return_proba and isinstance(model, ClassificationModel):
            probabilities = model.predict_proba(input_df)
            result['probabilities'] = probabilities.tolist()
            
        return result
        
    def get_model_info(self, model_id: int, version: Optional[str] = None) -> Dict[str, Any]:
        """Get model information and metadata"""
        # Get model metadata
        db_model = self.db.query(MLModel).filter_by(id=model_id).first()
        if not db_model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            
        # Get model version
        model_version = self.registry.get_model_version(model_id, version)
        if not model_version:
            raise HTTPException(status_code=404, detail="Model version not found")
            
        return {
            'model': {
                'id': db_model.id,
                'name': db_model.name,
                'type': db_model.type,
                'algorithm': db_model.algorithm,
                'created_at': db_model.created_at,
                'updated_at': db_model.updated_at
            },
            'version': {
                'version': model_version.version,
                'stage': model_version.stage,
                'status': model_version.status,
                'metrics': model_version.metrics,
                'parameters': model_version.parameters,
                'description': model_version.description,
                'created_at': model_version.created_at
            }
        }
        
    def get_model_metrics(self, model_id: int, version: Optional[str] = None) -> Dict[str, Any]:
        """Get model performance metrics"""
        model_version = self.registry.get_model_version(model_id, version)
        if not model_version:
            raise HTTPException(status_code=404, detail="Model version not found")
            
        return {
            'version': model_version.version,
            'metrics': model_version.metrics,
            'parameters': model_version.parameters
        }
