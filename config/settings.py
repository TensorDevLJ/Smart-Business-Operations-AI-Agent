"""
Application configuration using Pydantic Settings.
Loads from environment variables / .env file.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
import os


class Settings(BaseSettings):
    # Application
    app_name: str = "Smart Business Operations AI Agent"
    app_version: str = "1.0.0"
    debug: bool = True
    log_level: str = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api"

    # Database
    database_url: str = "sqlite:///./smart_business.db"

    # LLM (Ollama)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048

    # ML Models
    ml_models_dir: str = "./ml_models/trained"
    ml_data_dir: str = "./ml_models/data"

    # FAISS
    faiss_index_path: str = "./ml_models/faiss_index"
    business_docs_dir: str = "./docs/business_docs"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Alerts
    alert_anomaly_threshold: float = 0.15
    alert_revenue_drop_threshold: float = 0.10

    # Dashboard
    backend_url: str = "http://localhost:8000"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton instance
settings = Settings()

# Ensure model directories exist
Path(settings.ml_models_dir).mkdir(parents=True, exist_ok=True)
Path(settings.ml_data_dir).mkdir(parents=True, exist_ok=True)
