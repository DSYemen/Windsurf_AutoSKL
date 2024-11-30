import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from app.services.model_trainer import ModelTrainer

@pytest.fixture
def classification_data():
    """Create sample classification data"""
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=2,
        random_state=42
    )
    return X, y

@pytest.fixture
def regression_data():
    """Create sample regression data"""
    X, y = make_regression(
        n_samples=100,
        n_features=5,
        n_informative=3,
        random_state=42
    )
    return X, y

def test_detect_task_type(classification_data, regression_data):
    """Test task type detection"""
    trainer = ModelTrainer()
    
    X_clf, y_clf = classification_data
    assert trainer._detect_task_type(y_clf) == 'classification'
    
    X_reg, y_reg = regression_data
    assert trainer._detect_task_type(y_reg) == 'regression'

def test_train_classification(classification_data):
    """Test model training for classification"""
    X, y = classification_data
    trainer = ModelTrainer()
    
    results = trainer.train(X, y, n_trials=10)
    assert isinstance(results, dict)
    assert 'model_type' in results
    assert 'score' in results
    assert results['score'] > 0.5

def test_train_regression(regression_data):
    """Test model training for regression"""
    X, y = regression_data
    trainer = ModelTrainer()
    
    results = trainer.train(X, y, n_trials=10)
    assert isinstance(results, dict)
    assert 'model_type' in results
    assert 'score' in results
    assert results['score'] > 0

def test_predict(classification_data):
    """Test model predictions"""
    X, y = classification_data
    trainer = ModelTrainer()
    
    trainer.train(X, y, n_trials=10)
    predictions = trainer.predict(X)
    
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == len(X)
    assert all(isinstance(p, (int, np.integer)) for p in predictions)

def test_save_load_model(tmp_path, classification_data):
    """Test model saving and loading"""
    X, y = classification_data
    trainer = ModelTrainer()
    
    # Train and save model
    trainer.train(X, y, n_trials=10)
    model_path = tmp_path / "model.joblib"
    trainer.save_model(str(model_path))
    
    # Load model and make predictions
    loaded_trainer = ModelTrainer.load_model(str(model_path))
    predictions = loaded_trainer.predict(X)
    
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == len(X)
