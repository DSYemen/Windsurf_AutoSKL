from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "AutoSKL"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./autoskl.db")
    
    # Redis settings
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    
    # Model settings
    MODEL_PATH: str = "models"
    DEFAULT_MODEL_TIMEOUT: int = 3600  # 1 hour
    
    # Training Configuration
    DEFAULT_N_TRIALS: int = 30 #100
    DEFAULT_EARLY_STOPPING_ROUNDS: int = 10 #20
    DEFAULT_CV_FOLDS: int = 5
    DEFAULT_TIMEOUT_MINUTES: int = 30
    MIN_SAMPLES_FOR_MODEL: int = 50
    MAX_FEATURES_FOR_MODEL: int = 1000

    # Model-specific thresholds
    SVM_MAX_SAMPLES: int = 10000
    KNN_MAX_FEATURES: int = 100
    NEURAL_NET_MIN_SAMPLES: int = 1000
    
    # Monitoring settings
    ENABLE_MONITORING: bool = True
    DRIFT_THRESHOLD: float = 0.1
    
    class Config:
        case_sensitive = True

settings = Settings()
