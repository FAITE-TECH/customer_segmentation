import os
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    EMAIL_USER: str | None = None
    EMAIL_PASS: str | None = None
    MODELS_DIR: str = str(Path(__file__).parent/"models")
    SCALER_FILE: str = "rfm_scaler.pkl"
    KMEANS_FILE: str = "rfm_kmeans.pkl"

    class Config:
        env_file = ".env"

settings= Settings()