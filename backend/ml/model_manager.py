"""
Model Manager — Singleton that manages all ML models.

Benefits:
- Models loaded once at startup (not on every request)
- Centralized model registry
- Easy to add new models
- Thread-safe model access
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from typing import Optional
from loguru import logger

from backend.ml.forecasting import SalesForecastingModel
from backend.ml.anomaly import AnomalyDetectionModel
from config.settings import settings


class ModelManager:
    """
    Singleton registry for all ML models.
    Load once, reuse everywhere.
    """
    _instance: Optional["ModelManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._forecasting_model: Optional[SalesForecastingModel] = None
        self._anomaly_model: Optional[AnomalyDetectionModel] = None
        self._initialized = True

    @property
    def forecasting(self) -> SalesForecastingModel:
        if self._forecasting_model is None:
            logger.info("Loading forecasting model...")
            self._forecasting_model = SalesForecastingModel(settings.ml_models_dir)
            try:
                self._forecasting_model.load()
                logger.success("Forecasting model loaded from disk")
            except FileNotFoundError:
                logger.warning("Forecasting model not found — will train on first use")
        return self._forecasting_model

    @property
    def anomaly(self) -> AnomalyDetectionModel:
        if self._anomaly_model is None:
            logger.info("Loading anomaly model...")
            self._anomaly_model = AnomalyDetectionModel(settings.ml_models_dir)
            try:
                self._anomaly_model.load()
                logger.success("Anomaly model loaded from disk")
            except FileNotFoundError:
                logger.warning("Anomaly model not found — will train on first use")
        return self._anomaly_model

    def reload_all(self):
        """Force reload all models from disk."""
        self._forecasting_model = None
        self._anomaly_model = None
        logger.info("All models cleared. Will reload on next access.")

    def status(self) -> dict:
        """Return health status of all models."""
        return {
            "forecasting": self.forecasting.get_model_info(),
            "anomaly": self.anomaly.get_model_info(),
        }


# Global singleton
model_manager = ModelManager()
