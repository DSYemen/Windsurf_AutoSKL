from sqlalchemy import Column, String, Integer, JSON, ForeignKey
from sqlalchemy.orm import relationship
from .base_model import BaseModel

class MLModel(BaseModel):
    __tablename__ = "ml_models"

    name = Column(String)
    type = Column(String)  # classification, regression, clustering, etc.
    algorithm = Column(String)
    parameters = Column(JSON)
    metrics = Column(JSON)
    feature_importance = Column(JSON)
    status = Column(String)  # training, completed, failed
    
    # Foreign keys
    user_id = Column(Integer, ForeignKey("users.id"))
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    
    # Relationships
    user = relationship("User", back_populates="ml_models")
    dataset = relationship("Dataset", back_populates="ml_models")
    training_logs = relationship("TrainingLog", back_populates="ml_model")
