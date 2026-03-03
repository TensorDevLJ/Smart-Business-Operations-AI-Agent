"""
Centralized logging configuration using Loguru.
Provides structured, leveled logging to both console and file.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from loguru import logger
from pathlib import Path

from config.settings import settings


def setup_logging():
    """Configure Loguru for the application."""
    # Remove default handler
    logger.remove()

    # Console handler (color-coded, human-readable)
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=settings.log_level,
        colorize=True,
    )

    # File handler (JSON-formatted for log aggregators)
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)

    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="00:00",      # New file every day
        retention="30 days",   # Keep 30 days of logs
        compression="zip",
    )

    # Separate error log
    logger.add(
        log_dir / "errors_{time:YYYY-MM-DD}.log",
        level="ERROR",
        rotation="00:00",
        retention="90 days",
    )

    logger.info(f"Logging initialized (level: {settings.log_level})")


# Run setup when module is imported
setup_logging()
