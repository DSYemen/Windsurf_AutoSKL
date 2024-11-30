# AutoSKL - Automated Machine Learning Framework

AutoSKL is a comprehensive automated machine learning framework built with FastAPI and scikit-learn. It provides end-to-end machine learning capabilities from data processing to model deployment and monitoring.

## 🌟 Features

### Automated Machine Learning
- Automatic task type detection (classification, regression)
- Intelligent algorithm selection
- Hyperparameter optimization using Optuna
- Support for multiple ML libraries (scikit-learn, XGBoost, LightGBM)

### Data Processing
- Automated data cleaning and preprocessing
- Support for multiple data formats (CSV, Excel, Parquet)
- Advanced feature engineering
- Data validation and quality checks

### Model Management
- Automated model training and evaluation
- Model versioning and tracking
- Continuous monitoring and drift detection
- Automatic model updates

### Visualization & Reporting
- Interactive dashboard built with Streamlit
- Comprehensive performance reports
- Data drift analysis visualization
- Feature importance plots

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/autoskl.git
cd autoskl
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Services

1. Start Redis server (required for background tasks):
```bash
redis-server
```

2. Start Celery worker:
```bash
celery -A app.services.auto_updater worker --loglevel=info
```

3. Start the API server:
```bash
uvicorn app.main:app --reload
```

4. Launch the dashboard:
```bash
streamlit run app/ui/dashboard.py
```

## 📚 Documentation

### API Endpoints

#### Training
- `POST /train`: Train a new model
  ```json
  {
    "file": "training_data.csv",
    "target_column": "target",
    "model_types": ["random_forest", "xgboost"],
    "n_trials": 100
  }
  ```

#### Predictions
- `POST /predict`: Make predictions
  ```json
  {
    "data": [
      {"feature1": 1.0, "feature2": "A"},
      {"feature1": 2.0, "feature2": "B"}
    ]
  }
  ```

#### Monitoring
- `GET /model/status`: Get model status
- `GET /reports`: List available reports

### Command Line Interface

```bash
# Train a model
python -m autoskl train --data data.csv --target target

# Make predictions
python -m autoskl predict --data new_data.csv

# Generate report
python -m autoskl report --model-id latest
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Generate coverage report:
```bash
pytest --cov=app tests/
```

## 📊 Dashboard

Access the interactive dashboard at `http://localhost:8501` after starting the Streamlit server.

Features:
- Model training interface
- Real-time predictions
- Performance monitoring
- Data drift analysis
- Report visualization

## 🛠️ Configuration

### Environment Variables
Create a `.env` file:
```env
DATABASE_URL=sqlite:///./autoskl.db
REDIS_HOST=localhost
REDIS_PORT=6379
MODEL_PATH=models
ENABLE_MONITORING=true
```

### Monitoring Configuration
Adjust monitoring settings in `config.py`:
```python
DRIFT_THRESHOLD = 0.1
UPDATE_FREQUENCY = "1d"
MIN_SAMPLES_REQUIRED = 1000
```

## 📚 Documentation

For detailed information about AutoSKL, please refer to the following documentation:

- [User Guide](docs/user_guide.md): Complete guide for using AutoSKL
- [API Reference](docs/api_reference.md): Detailed API documentation
- [Development Guide](docs/development_guide.md): Guide for developers

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- scikit-learn team
- FastAPI
- Streamlit
- Optuna
