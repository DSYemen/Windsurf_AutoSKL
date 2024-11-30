import pytest
import pandas as pd
import numpy as np
from app.services.validator import DataValidator, DataValidationError

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return pd.DataFrame({
        'numeric': [1, 2, np.nan, 4, 5] * 20,
        'categorical': ['A', 'B', 'A', None, 'C'] * 20,
        'many_missing': [1, np.nan, np.nan, np.nan, 2] * 20,
        'unique': range(100),
        'target': [0, 1, 1, 0, 1] * 20
    })

def test_check_missing_values(sample_data):
    """Test missing values detection"""
    validator = DataValidator(max_missing_ratio=0.2)
    missing_values = validator._check_missing_values(sample_data)
    
    assert isinstance(missing_values, dict)
    assert missing_values['numeric'] == 20
    assert missing_values['categorical'] == 20
    assert missing_values['many_missing'] == 60
    assert len(validator.validation_errors) > 0

def test_check_data_types(sample_data):
    """Test data type checking"""
    validator = DataValidator()
    data_types = validator._check_data_types(sample_data)
    
    assert isinstance(data_types, dict)
    assert 'numeric' in data_types
    assert 'categorical' in data_types
    assert len(data_types) == 5

def test_check_unique_values(sample_data):
    """Test unique values checking"""
    validator = DataValidator()
    unique_values = validator._check_unique_values(sample_data)
    
    assert isinstance(unique_values, dict)
    assert unique_values['categorical'] == 3
    assert unique_values['unique'] == 100
    assert len(validator.warnings) > 0

def test_validate_training_data(sample_data):
    """Test training data validation"""
    validator = DataValidator()
    report = validator.validate_training_data(sample_data, 'target')
    
    assert report.total_rows == len(sample_data)
    assert report.total_columns == len(sample_data.columns)
    assert len(report.validation_errors) > 0
    assert len(report.warnings) > 0

def test_validate_prediction_data(sample_data):
    """Test prediction data validation"""
    validator = DataValidator()
    training_columns = ['numeric', 'categorical']
    
    # Create prediction data with missing and extra columns
    pred_data = pd.DataFrame({
        'numeric': [1, 2, 3],
        'categorical': ['A', 'B', 'C'],
        'extra_col': [1, 2, 3]
    })
    
    report = validator.validate_prediction_data(pred_data, training_columns)
    assert len(report.warnings) > 0  # Should warn about extra column

def test_validation_errors():
    """Test validation error handling"""
    validator = DataValidator()
    
    # Test empty dataset
    with pytest.raises(DataValidationError):
        validator.validate_training_data(pd.DataFrame(), 'target')
        
    # Test missing target column
    data = pd.DataFrame({'feature': [1, 2, 3]})
    with pytest.raises(DataValidationError):
        validator.validate_training_data(data, 'target')
