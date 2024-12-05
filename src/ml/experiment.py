from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import pandas as pd
from sqlalchemy import Column, Integer, String, JSON, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from src.models.base_model import BaseModel

class Experiment(BaseModel):
    __tablename__ = "experiments"

    name = Column(String)
    description = Column(String)
    status = Column(String)  # running, completed, failed
    parameters = Column(JSON)
    metrics = Column(JSON)
    artifacts = Column(JSON)  # paths to saved models, plots, etc.
    
    # Foreign keys
    user_id = Column(Integer, ForeignKey("users.id"))
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    
    # Relationships
    user = relationship("User", back_populates="experiments")
    dataset = relationship("Dataset")
    runs = relationship("ExperimentRun", back_populates="experiment")

class ExperimentRun(BaseModel):
    __tablename__ = "experiment_runs"

    run_number = Column(Integer)
    status = Column(String)
    parameters = Column(JSON)
    metrics = Column(JSON)
    artifacts = Column(JSON)
    
    # Foreign key
    experiment_id = Column(Integer, ForeignKey("experiments.id"))
    
    # Relationship
    experiment = relationship("Experiment", back_populates="runs")

class ExperimentTracker:
    def __init__(self, db_session, user_id: int):
        self.db = db_session
        self.user_id = user_id
        self.current_experiment = None
        self.current_run = None
        
    def create_experiment(
        self,
        name: str,
        description: str,
        dataset_id: int,
        parameters: Dict[str, Any]
    ) -> Experiment:
        """Create a new experiment"""
        experiment = Experiment(
            name=name,
            description=description,
            status="created",
            parameters=parameters,
            user_id=self.user_id,
            dataset_id=dataset_id
        )
        
        self.db.add(experiment)
        self.db.commit()
        self.current_experiment = experiment
        return experiment
        
    def start_run(self, parameters: Dict[str, Any]) -> ExperimentRun:
        """Start a new run for the current experiment"""
        if not self.current_experiment:
            raise ValueError("No active experiment")
            
        # Get next run number
        run_number = len(self.current_experiment.runs) + 1
        
        run = ExperimentRun(
            experiment_id=self.current_experiment.id,
            run_number=run_number,
            status="running",
            parameters=parameters
        )
        
        self.db.add(run)
        self.db.commit()
        self.current_run = run
        return run
        
    def log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics for current run"""
        if not self.current_run:
            raise ValueError("No active run")
            
        self.current_run.metrics = metrics
        self.db.commit()
        
    def log_artifact(self, name: str, path: str):
        """Log artifact path for current run"""
        if not self.current_run:
            raise ValueError("No active run")
            
        artifacts = self.current_run.artifacts or {}
        artifacts[name] = path
        self.current_run.artifacts = artifacts
        self.db.commit()
        
    def end_run(self, status: str = "completed"):
        """End current run"""
        if not self.current_run:
            raise ValueError("No active run")
            
        self.current_run.status = status
        self.db.commit()
        self.current_run = None
        
    def end_experiment(self, status: str = "completed"):
        """End current experiment"""
        if not self.current_experiment:
            raise ValueError("No active experiment")
            
        self.current_experiment.status = status
        self.db.commit()
        self.current_experiment = None
        
    def get_experiment_summary(self, experiment_id: int) -> Dict[str, Any]:
        """Get summary of experiment runs"""
        experiment = self.db.query(Experiment).filter_by(id=experiment_id).first()
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        runs_df = pd.DataFrame([
            {
                "run_number": run.run_number,
                "status": run.status,
                **run.parameters,
                **(run.metrics or {})
            }
            for run in experiment.runs
        ])
        
        return {
            "experiment": {
                "name": experiment.name,
                "description": experiment.description,
                "status": experiment.status,
                "created_at": experiment.created_at,
                "parameters": experiment.parameters
            },
            "runs": runs_df.to_dict(orient="records"),
            "best_run": runs_df.loc[runs_df["metrics"].idxmax()] if len(runs_df) > 0 else None
        }
        
    def get_run_details(self, run_id: int) -> Dict[str, Any]:
        """Get detailed information about a specific run"""
        run = self.db.query(ExperimentRun).filter_by(id=run_id).first()
        if not run:
            raise ValueError(f"Run {run_id} not found")
            
        return {
            "run_number": run.run_number,
            "status": run.status,
            "parameters": run.parameters,
            "metrics": run.metrics,
            "artifacts": run.artifacts,
            "created_at": run.created_at,
            "updated_at": run.updated_at
        }
