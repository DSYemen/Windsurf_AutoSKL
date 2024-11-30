# AutoSKL API Reference

## API Endpoints

### Training

#### Train Model
```http
POST /train
```

Train a new machine learning model.

**Request Body**
- `file`: Training data file (multipart/form-data)
- `target_column`: Name of target column
- `model_types` (optional): List of model types to try
- `n_trials` (optional): Number of optimization trials

**Response**
```json
{
    "status": "success",
    "training_results": {
        "model_type": "random_forest",
        "score": 0.95,
        "parameters": {}
    },
    "report_path": "reports/training_report_20231120_123456.html"
}
```

### Predictions

#### Make Predictions
```http
POST /predict
```

Make predictions using trained model.

**Request Body**
```json
{
    "data": [
        {"feature1": 1.0, "feature2": "A"},
        {"feature1": 2.0, "feature2": "B"}
    ]
}
```

**Response**
```json
{
    "status": "success",
    "predictions": [0, 1],
    "monitoring": {
        "drift_analysis": {},
        "predictions": [],
        "timestamp": "2023-11-20T12:34:56"
    }
}
```

### Model Status

#### Get Model Status
```http
GET /model/status
```

Get current model status and monitoring information.

**Response**
```json
{
    "model_available": true,
    "monitoring_enabled": true,
    "update_status": {
        "current_version": 1,
        "last_update": "2023-11-20T12:34:56",
        "performance_history": []
    }
}
```

### Reports

#### List Reports
```http
GET /reports
```

Get list of available reports.

**Response**
```json
{
    "reports": [
        {
            "name": "training_report_20231120_123456",
            "path": "reports/training_report_20231120_123456.html",
            "type": "html",
            "created": "2023-11-20T12:34:56"
        }
    ]
}
```

## Python Client

### Installation
```bash
pip install autoskl-client
```

### Usage

```python
from autoskl.client import AutoSKLClient

# Initialize client
client = AutoSKLClient('http://localhost:8000')

# Train model
with open('data.csv', 'rb') as f:
    result = client.train(
        data=f,
        target_column='target',
        model_types=['random_forest', 'xgboost']
    )

# Make predictions
predictions = client.predict([
    {'feature1': 1.0, 'feature2': 'A'},
    {'feature1': 2.0, 'feature2': 'B'}
])

# Get model status
status = client.get_status()

# List reports
reports = client.list_reports()
```

## Command Line Interface

### Training
```bash
python -m autoskl train --data data.csv --target target
```

Options:
- `--data`: Path to training data file
- `--target`: Target column name
- `--model-types`: Model types to try
- `--n-trials`: Number of optimization trials
- `--output`: Output directory for model

### Predictions
```bash
python -m autoskl predict --data new_data.csv
```

Options:
- `--data`: Path to input data file
- `--model`: Path to model file
- `--output`: Path to save predictions

### Reports
```bash
python -m autoskl report --model-id latest
```

Options:
- `--model-id`: Model ID or "latest"
- `--type`: Report type (training/monitoring)
- `--output`: Output directory for report

### Server
```bash
python -m autoskl serve
```

Options:
- `--host`: API server host
- `--port`: API server port
- `--reload`: Enable auto-reload

### Dashboard
```bash
python -m autoskl dashboard
```

Options:
- `--port`: Dashboard port

## Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| 400  | Bad Request | Check request parameters |
| 404  | Not Found   | Ensure model is trained |
| 500  | Server Error| Check server logs |

## Rate Limits

- Training: 10 requests per hour
- Predictions: 1000 requests per minute
- Status: 100 requests per minute
- Reports: 50 requests per minute

## Security

### Authentication
Bearer token authentication required for all endpoints.

Header format:
```http
Authorization: Bearer <token>
```

### Access Control
- Read-only access: Predictions, Status, Reports
- Write access: Training, Model Updates
- Admin access: All endpoints

### Data Security
- TLS encryption for all API calls
- Data validation and sanitization
- Audit logging of all operations
