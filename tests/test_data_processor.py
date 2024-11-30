import pytest
import pandas as pd
import numpy as np
from app.services.data_processor import DataProcessor

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return pd.DataFrame({
        'numeric': [1, 2, np.nan, 4, 5],
        'categorical': ['A', 'B', 'A', None, 'C'],
        'target': [0, 1, 1, 0, 1]
    })

def test_identify_column_types(sample_data):
    """Test column type identification"""
    processor = DataProcessor()
    num_cols, cat_cols = processor._identify_column_types(sample_data)
    
    assert 'numeric' in num_cols
    assert 'categorical' in cat_cols
    assert len(num_cols) == 1
    assert len(cat_cols) == 2

def test_fit_transform(sample_data):
    """Test data transformation"""
    processor = DataProcessor()
    X, y = processor.fit_transform(sample_data, 'target')
    
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == len(sample_data)
    assert y.shape[0] == len(sample_data)
    assert not np.isnan(X).any()

def test_transform(sample_data):
    """Test transform without target"""
    processor = DataProcessor()
    processor.fit_transform(sample_data, 'target')
    
    new_data = pd.DataFrame({
        'numeric': [3, np.nan],
        'categorical': ['B', 'A']
    })
    
    X = processor.transform(new_data)
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == len(new_data)
    assert not np.isnan(X).any()
