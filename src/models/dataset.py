from sqlalchemy import Column, String, Integer, JSON, ForeignKey
from sqlalchemy.orm import relationship
from .base_model import BaseModel

class Dataset(BaseModel):
    __tablename__ = "datasets"

    name = Column(String)
    description = Column(String)
    file_path = Column(String)
    file_type = Column(String)
    features = Column(JSON)  # Column metadata
    target = Column(String)
    preprocessing_steps = Column(JSON)
    statistics = Column(JSON)  # Basic statistics about the dataset
    
    # Foreign keys
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    user = relationship("User", back_populates="datasets")
    ml_models = relationship("MLModel", back_populates="dataset")
