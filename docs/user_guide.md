# AutoSKL User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [Data Preparation](#data-preparation)
5. [Model Training](#model-training)
6. [Making Predictions](#making-predictions)
7. [Monitoring & Maintenance](#monitoring--maintenance)
8. [Troubleshooting](#troubleshooting)

## Introduction

AutoSKL is an automated machine learning framework designed to simplify the process of building, deploying, and maintaining machine learning models. It provides a comprehensive set of tools for data preprocessing, model training, prediction, and monitoring.

### Key Features
- Automated model selection and training
- Advanced data preprocessing
- Continuous model monitoring
- Interactive dashboard
- Command-line interface
- RESTful API

## Installation

### Prerequisites
- Python 3.8 or higher
- Redis server (for background tasks)
- Git (optional)

### Step-by-Step Installation
1. Clone the repository or download the source code
2. Create a virtual environment
3. Install dependencies
4. Configure environment variables

Detailed instructions in [README.md](../README.md)

## Getting Started

### Quick Start
1. Start the services:
```bash
# Start Redis
redis-server

# Start Celery worker
celery -A app.services.auto_updater worker --loglevel=info

# Start API server
python -m autoskl serve

# Launch dashboard
python -m autoskl dashboard
```

2. Access the interfaces:
- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs

## Data Preparation

### Supported Formats
- CSV files
- Excel files (.xlsx, .xls)
- Parquet files

### Data Requirements
- Clean column names (no special characters)
- Consistent data types
- Target column clearly identified
- No duplicate rows

### Data Validation
The system automatically checks for:
- Missing values
- Data type consistency
- Unique value ratios
- Outliers

### Example Dataset Structure
```csv
id,feature1,feature2,target
1,23.5,category_a,0
2,45.2,category_b,1
3,67.8,category_a,0
```

## Model Training

### Using the Dashboard
1. Navigate to "Train Model"
2. Upload training data
3. Select target column
4. Configure training parameters
5. Click "Train Model"

### Using CLI
```bash
python -m autoskl train --data data.csv --target target
```

### Using API
```python
import requests

files = {'file': open('data.csv', 'rb')}
data = {'target_column': 'target', 'n_trials': 100}
response = requests.post('http://localhost:8000/train', files=files, data=data)
```

### Training Parameters
- `model_types`: List of algorithms to try
- `n_trials`: Number of optimization trials
- `target_column`: Name of target variable

## Making Predictions

### Using the Dashboard
1. Navigate to "Make Predictions"
2. Upload new data
3. Click "Make Predictions"
4. View or download results

### Using CLI
```bash
python -m autoskl predict --data new_data.csv
```

### Using API
```python
import requests

data = {'data': [{'feature1': 23.5, 'feature2': 'category_a'}]}
response = requests.post('http://localhost:8000/predict', json=data)
```

## Monitoring & Maintenance

### Model Performance
The system tracks:
- Prediction accuracy
- Data drift
- Model staleness
- Resource usage

### Automatic Updates
The system can:
- Detect when retraining is needed
- Automatically retrain models
- Maintain model versions
- Generate performance reports

### Monitoring Dashboard
Access monitoring features:
1. View current model status
2. Check performance metrics
3. Analyze data drift
4. Review update history

## Troubleshooting

### Common Issues

#### Installation Problems
- **Issue**: Dependencies installation fails
  - **Solution**: Update pip and try installing dependencies one by one
  
- **Issue**: Redis connection error
  - **Solution**: Ensure Redis server is running

#### Training Problems
- **Issue**: Out of memory
  - **Solution**: Reduce dataset size or use batch processing
  
- **Issue**: Training too slow
  - **Solution**: Reduce n_trials or limit model types

#### Prediction Problems
- **Issue**: Mismatched features
  - **Solution**: Ensure prediction data matches training data structure
  
- **Issue**: Unexpected results
  - **Solution**: Check data preprocessing and model status

### Getting Help
- Check the [FAQ](faq.md)
- Review error messages
- Check system logs
- Contact support team

## Advanced Topics

### Custom Models
Add custom models by:
1. Implementing model interface
2. Registering model with system
3. Using in training

### Feature Engineering
Available transformations:
- Numeric scaling
- Categorical encoding
- Feature selection
- Custom transformers

### API Integration
Integration examples:
- Python clients
- REST API calls
- Batch processing
- Streaming predictions

### Security
Security features:
- API authentication
- Data encryption
- Access control
- Audit logging
