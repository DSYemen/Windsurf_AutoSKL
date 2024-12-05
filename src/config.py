import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    BASE_DIR: Path = Path(__file__).parent.parent
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/autoskl.db"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REDIS_URL: str = "redis://localhost:6379"
    MODEL_STORE_PATH: Path = BASE_DIR / "models"
    UPLOAD_FOLDER: Path = BASE_DIR / "uploads"
    EXPORT_FOLDER: Path = BASE_DIR / "exports"
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"

settings = Settings()
