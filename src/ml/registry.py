from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
from sqlalchemy import Column, Integer, String, JSON, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import relationship
from src.models.base_model import BaseModel
from src.config import settings

class ModelVersion(BaseModel):
    __tablename__ = "model_versions"

    version = Column(String)
    stage = Column(String)  # dev, staging, production
    status = Column(String)  # active, archived
    metrics = Column(JSON)
    parameters = Column(JSON)
    description = Column(String)
    is_current = Column(Boolean, default=False)
    
    # Foreign keys
    model_id = Column(Integer, ForeignKey("ml_models.id"))
    experiment_run_id = Column(Integer, ForeignKey("experiment_runs.id"))
    
    # Relationships
    model = relationship("MLModel", back_populates="versions")
    experiment_run = relationship("ExperimentRun")

class ModelRegistry:
    def __init__(self, db_session):
        self.db = db_session
        self.registry_path = settings.MODEL_STORE_PATH / "registry"
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
    def _get_version_path(self, model_id: int, version: str) -> Path:
        """Get path for model version artifacts"""
        return self.registry_path / f"model_{model_id}" / f"version_{version}"
        
    def register_model(
        self,
        model_id: int,
        experiment_run_id: int,
        metrics: Dict[str, Any],
        parameters: Dict[str, Any],
        description: str,
        model_files: Dict[str, Path]
    ) -> ModelVersion:
        """Register a new model version"""
        # Get current version number
        versions = self.db.query(ModelVersion).filter_by(model_id=model_id).all()
        new_version = f"v{len(versions) + 1}"
        
        # Create version record
        version = ModelVersion(
            model_id=model_id,
            experiment_run_id=experiment_run_id,
            version=new_version,
            stage="dev",
            status="active",
            metrics=metrics,
            parameters=parameters,
            description=description
        )
        
        # Set as current version if first version
        if len(versions) == 0:
            version.is_current = True
        
        self.db.add(version)
        self.db.commit()
        
        # Save model files
        version_path = self._get_version_path(model_id, new_version)
        version_path.mkdir(parents=True, exist_ok=True)
        
        for name, file_path in model_files.items():
            dest_path = version_path / name
            shutil.copy2(file_path, dest_path)
        
        return version
        
    def transition_stage(
        self,
        model_id: int,
        version: str,
        new_stage: str
    ) -> ModelVersion:
        """Transition model version to new stage"""
        valid_stages = ["dev", "staging", "production"]
        if new_stage not in valid_stages:
            raise ValueError(f"Invalid stage. Must be one of {valid_stages}")
            
        version_record = self.db.query(ModelVersion).filter_by(
            model_id=model_id,
            version=version
        ).first()
        
        if not version_record:
            raise ValueError(f"Version {version} not found for model {model_id}")
            
        # If transitioning to production, archive current production version
        if new_stage == "production":
            current_prod = self.db.query(ModelVersion).filter_by(
                model_id=model_id,
                stage="production",
                status="active"
            ).first()
            
            if current_prod:
                current_prod.stage = "archived"
                current_prod.status = "archived"
                current_prod.is_current = False
        
        version_record.stage = new_stage
        if new_stage == "production":
            version_record.is_current = True
        
        self.db.commit()
        return version_record
        
    def get_model_version(
        self,
        model_id: int,
        version: Optional[str] = None
    ) -> Optional[ModelVersion]:
        """Get specific model version or current version"""
        query = self.db.query(ModelVersion).filter_by(model_id=model_id)
        
        if version:
            return query.filter_by(version=version).first()
        return query.filter_by(is_current=True).first()
        
    def get_model_files(
        self,
        model_id: int,
        version: str
    ) -> Dict[str, Path]:
        """Get paths to model version files"""
        version_path = self._get_version_path(model_id, version)
        if not version_path.exists():
            raise ValueError(f"Files for version {version} not found")
            
        return {
            file.name: file
            for file in version_path.iterdir()
            if file.is_file()
        }
        
    def compare_versions(
        self,
        model_id: int,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """Compare two model versions"""
        v1 = self.get_model_version(model_id, version1)
        v2 = self.get_model_version(model_id, version2)
        
        if not v1 or not v2:
            raise ValueError("One or both versions not found")
            
        # Compare metrics
        metric_diff = {
            k: {
                'v1': v1.metrics.get(k),
                'v2': v2.metrics.get(k),
                'diff': v2.metrics.get(k) - v1.metrics.get(k)
                if k in v1.metrics and k in v2.metrics
                else None
            }
            for k in set(v1.metrics) | set(v2.metrics)
        }
        
        # Compare parameters
        param_diff = {
            k: {
                'v1': v1.parameters.get(k),
                'v2': v2.parameters.get(k),
                'changed': v1.parameters.get(k) != v2.parameters.get(k)
            }
            for k in set(v1.parameters) | set(v2.parameters)
        }
        
        return {
            'metrics_comparison': metric_diff,
            'parameter_comparison': param_diff,
            'version1': {
                'version': v1.version,
                'stage': v1.stage,
                'created_at': v1.created_at,
                'description': v1.description
            },
            'version2': {
                'version': v2.version,
                'stage': v2.stage,
                'created_at': v2.created_at,
                'description': v2.description
            }
        }
        
    def list_versions(
        self,
        model_id: int,
        stage: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all versions for a model"""
        query = self.db.query(ModelVersion).filter_by(model_id=model_id)
        if stage:
            query = query.filter_by(stage=stage)
            
        versions = query.all()
        return [
            {
                'version': v.version,
                'stage': v.stage,
                'status': v.status,
                'is_current': v.is_current,
                'created_at': v.created_at,
                'metrics': v.metrics,
                'description': v.description
            }
            for v in versions
        ]
