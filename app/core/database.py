from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import Dict, Any, Optional, Union
import json
import joblib
import io
from ..core.config import settings
import pandas as pd

Base = declarative_base()

class MLModel(Base):
    __tablename__ = "ml_models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    version = Column(String)
    model_type = Column(String)  # classification/regression
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    hyperparameters = Column(JSON)
    metrics = Column(JSON)
    feature_importance = Column(JSON, nullable=True)
    model_binary = Column(LargeBinary)
    preprocessing_params = Column(JSON)
    
class ModelTrainingLog(Base):
    __tablename__ = "model_training_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    event_type = Column(String)  # training_started, training_completed, error
    details = Column(JSON)

class ModelPerformanceLog(Base):
    __tablename__ = "model_performance_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metrics = Column(JSON)

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    prediction = Column(Float)
    actual = Column(Float, nullable=True)
    features = Column(JSON)

class ResourceUsageLog(Base):
    __tablename__ = "resource_usage_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    memory_usage = Column(Float)  # MB
    cpu_usage = Column(Float)  # Percentage
    disk_usage = Column(Float)  # Percentage

class DatabaseManager:
    def __init__(self):
        self.engine = create_engine(settings.DATABASE_URL)
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def save_model(self, 
                  name: str,
                  model: Any,
                  model_type: str,
                  hyperparameters: Dict,
                  metrics: Dict,
                  feature_importance: Optional[Dict] = None,
                  preprocessing_params: Optional[Dict] = None) -> int:
        """Save trained model to database"""
        # Serialize model
        model_buffer = io.BytesIO()
        joblib.dump(model, model_buffer)
        model_binary = model_buffer.getvalue()
        
        db = self.SessionLocal()
        try:
            # Create new model version
            existing_models = db.query(MLModel).filter(MLModel.name == name).count()
            version = f"v{existing_models + 1}"
            
            ml_model = MLModel(
                name=name,
                version=version,
                model_type=model_type,
                hyperparameters=hyperparameters,
                metrics=metrics,
                feature_importance=feature_importance,
                model_binary=model_binary,
                preprocessing_params=preprocessing_params
            )
            
            db.add(ml_model)
            db.commit()
            db.refresh(ml_model)
            return ml_model.id
        finally:
            db.close()
    
    def get_model_id_by_name(self, model_name: str) -> Optional[int]:
        """Get model ID by name (returns latest version)"""
        db = self.SessionLocal()
        try:
            model = db.query(MLModel)\
                     .filter(MLModel.name == model_name)\
                     .order_by(MLModel.version.desc())\
                     .first()
            return model.id if model else None
        finally:
            db.close()
            
    def load_model(self, model_identifier: Union[int, str]) -> Optional[Dict[str, Any]]:
        """Load model from database by ID or name"""
        db = self.SessionLocal()
        try:
            # If string provided, get ID by name
            if isinstance(model_identifier, str):
                model_id = self.get_model_id_by_name(model_identifier)
                if model_id is None:
                    return None
            else:
                model_id = model_identifier
                
            model = db.query(MLModel).filter(MLModel.id == model_id).first()
            if not model:
                return None
                
            # Deserialize model
            model_buffer = io.BytesIO(model.model_binary)
            loaded_model = joblib.load(model_buffer)
            
            return {
                'model': loaded_model,
                'hyperparameters': model.hyperparameters,
                'metrics': model.metrics,
                'feature_importance': model.feature_importance,
                'preprocessing_params': model.preprocessing_params,
                'type': model.model_type,
                'version': model.version,
                'created_at': model.created_at
            }
        finally:
            db.close()
    
    def log_training_event(self, model_id: int, event_type: str, details: Dict):
        """Log training events"""
        db = self.SessionLocal()
        try:
            log = ModelTrainingLog(
                model_id=model_id,
                event_type=event_type,
                details=details
            )
            db.add(log)
            db.commit()
        finally:
            db.close()
    
    def get_all_model_ids(self) -> list:
        """الحصول على قائمة معرفات جميع النماذج"""
        db = self.SessionLocal()
        try:
            models = db.query(MLModel.id).all()
            return [model[0] for model in models]
        finally:
            db.close()
            
    def get_all_model_names(self) -> list:
        """الحصول على قائمة أسماء جميع النماذج"""
        db = self.SessionLocal()
        try:
            models = db.query(MLModel.name).distinct().all()
            return [model[0] for model in models]
        finally:
            db.close()
            
    def get_model_history(self, model_name: str) -> list:
        """الحصول على تاريخ تدريب النموذج"""
        db = self.SessionLocal()
        try:
            models = db.query(MLModel).filter(
                MLModel.name == model_name
            ).order_by(MLModel.created_at.desc()).all()
            
            return [{
                'id': model.id,
                'version': model.version,
                'created_at': model.created_at,
                'metrics': model.metrics,
                'hyperparameters': model.hyperparameters
            } for model in models]
        finally:
            db.close()

    def get_all_models(self) -> list:
        """الحصول على جميع النماذج المحفوظة"""
        db = self.SessionLocal()
        try:
            models = db.query(MLModel).order_by(MLModel.created_at.desc()).all()
            return [{
                'id': model.id,
                'name': model.name,
                'version': model.version,
                'model_type': model.model_type,
                'created_at': model.created_at,
                'metrics': model.metrics,
                'hyperparameters': model.hyperparameters,
                'feature_importance': model.feature_importance,
                'preprocessing_params': model.preprocessing_params
            } for model in models]
        finally:
            db.close()

    def get_model_performance_history(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get model performance history"""
        db = self.SessionLocal()
        try:
            logs = db.query(ModelPerformanceLog).filter(
                ModelPerformanceLog.timestamp.between(start_date, end_date)
            ).order_by(ModelPerformanceLog.timestamp).all()
            
            if not logs:
                return pd.DataFrame()
                
            data = [{
                'timestamp': log.timestamp,
                'model_id': log.model_id,
                **log.metrics
            } for log in logs]
            
            return pd.DataFrame(data)
        finally:
            db.close()
            
    def get_prediction_history(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get prediction history"""
        db = self.SessionLocal()
        try:
            logs = db.query(PredictionLog).filter(
                PredictionLog.timestamp.between(start_date, end_date)
            ).order_by(PredictionLog.timestamp).all()
            
            if not logs:
                return pd.DataFrame()
                
            data = [{
                'timestamp': log.timestamp,
                'model_id': log.model_id,
                'prediction': log.prediction,
                'actual': log.actual,
                **log.features
            } for log in logs]
            
            return pd.DataFrame(data)
        finally:
            db.close()
            
    def get_resource_usage_history(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get resource usage history"""
        db = self.SessionLocal()
        try:
            logs = db.query(ResourceUsageLog).filter(
                ResourceUsageLog.timestamp.between(start_date, end_date)
            ).order_by(ResourceUsageLog.timestamp).all()
            
            if not logs:
                return pd.DataFrame()
                
            data = [{
                'timestamp': log.timestamp,
                'memory_usage': log.memory_usage,
                'cpu_usage': log.cpu_usage,
                'disk_usage': log.disk_usage
            } for log in logs]
            
            return pd.DataFrame(data)
        finally:
            db.close()
            
    def log_model_performance(self, model_id: int, metrics: Dict[str, float]):
        """Log model performance metrics"""
        db = self.SessionLocal()
        try:
            log = ModelPerformanceLog(
                model_id=model_id,
                metrics=metrics
            )
            db.add(log)
            db.commit()
        finally:
            db.close()
            
    def log_prediction(self, model_id: int, prediction: float, features: Dict[str, Any], actual: Optional[float] = None):
        """Log model prediction"""
        db = self.SessionLocal()
        try:
            log = PredictionLog(
                model_id=model_id,
                prediction=prediction,
                actual=actual,
                features=features
            )
            db.add(log)
            db.commit()
        finally:
            db.close()
            
    def log_resource_usage(self, memory_usage: float, cpu_usage: float, disk_usage: float):
        """Log system resource usage"""
        db = self.SessionLocal()
        try:
            log = ResourceUsageLog(
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                disk_usage=disk_usage
            )
            db.add(log)
            db.commit()
        finally:
            db.close()
