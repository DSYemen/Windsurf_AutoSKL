from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import Dict, Any, List
import pandas as pd

from src.core.security import get_current_user
from src.models.user import User
from src.models.ml_model import MLModel
from src.models.dataset import Dataset
from src.ml.classification import ClassificationModel
from src.ml.regression import RegressionModel
from src.database import get_db

router = APIRouter()

@router.post("/models/train")
async def train_model(
    model_type: str,
    algorithm: str,
    dataset_id: int,
    parameters: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Get dataset
    dataset = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.user_id == current_user.id
    ).first()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Load data
    data = pd.read_csv(dataset.file_path)
    X = data.drop(columns=[dataset.target])
    y = data[dataset.target]
    
    # Create and train model
    if model_type == "classification":
        model = ClassificationModel(
            model_name=f"{algorithm}_{dataset_id}",
            user_id=current_user.id,
            algorithm=algorithm,
            params=parameters
        )
    elif model_type == "regression":
        model = RegressionModel(
            model_name=f"{algorithm}_{dataset_id}",
            user_id=current_user.id,
            algorithm=algorithm,
            params=parameters
        )
    else:
        raise HTTPException(status_code=400, detail="Unsupported model type")
    
    # Train model
    model.train(X, y)
    
    # Save model metadata to database
    db_model = MLModel(
        name=model.model_name,
        type=model_type,
        algorithm=algorithm,
        parameters=parameters,
        metrics=model.evaluate(X, y),
        feature_importance=model.get_feature_importance(),
        status="completed",
        user_id=current_user.id,
        dataset_id=dataset_id
    )
    
    db.add(db_model)
    db.commit()
    
    # Save model to disk
    model.save_model()
    
    return {"message": "Model trained successfully", "model_id": db_model.id}

@router.post("/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    description: str = None,
    target_column: str = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Save file
    file_path = f"uploads/user_{current_user.id}/{file.filename}"
    data = pd.read_csv(file.file)
    
    # Calculate basic statistics
    statistics = {
        "rows": len(data),
        "columns": len(data.columns),
        "missing_values": data.isnull().sum().to_dict()
    }
    
    # Create dataset record
    dataset = Dataset(
        name=file.filename,
        description=description,
        file_path=file_path,
        file_type="csv",
        features=list(data.columns),
        target=target_column,
        statistics=statistics,
        user_id=current_user.id
    )
    
    db.add(dataset)
    db.commit()
    
    # Save file to disk
    data.to_csv(file_path, index=False)
    
    return {"message": "Dataset uploaded successfully", "dataset_id": dataset.id}

@router.get("/models/{model_id}/predictions")
async def get_predictions(
    model_id: int,
    input_data: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Get model from database
    db_model = db.query(MLModel).filter(
        MLModel.id == model_id,
        MLModel.user_id == current_user.id
    ).first()
    
    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Load model
    if db_model.type == "classification":
        model = ClassificationModel(
            model_name=db_model.name,
            user_id=current_user.id,
            algorithm=db_model.algorithm
        )
    elif db_model.type == "regression":
        model = RegressionModel(
            model_name=db_model.name,
            user_id=current_user.id,
            algorithm=db_model.algorithm
        )
    else:
        raise HTTPException(status_code=400, detail="Unsupported model type")
    
    if not model.load_model():
        raise HTTPException(status_code=404, detail="Model file not found")
    
    # Make prediction
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    
    return {"prediction": prediction.tolist()}
