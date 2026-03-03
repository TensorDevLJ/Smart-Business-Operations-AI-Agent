"""
FastAPI Application Entry Point.

Production-grade setup with:
- Lifespan events (startup/shutdown)
- CORS middleware
- Exception handlers
- Request logging
- Health checks
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from config.settings import settings
from backend.database.connection import init_db
from backend.utils.logger import setup_logging


# ─────────────────────────────────────────────
#  Lifespan (startup / shutdown)
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    # ── Startup ──────────────────────────────────
    logger.info("🚀 Starting Smart Business AI Agent...")

    # Initialize database
    init_db()

    # Pre-load ML models (warm-up)
    try:
        from backend.ml.model_manager import model_manager
        _ = model_manager.forecasting
        _ = model_manager.anomaly
        logger.success("✅ ML models pre-loaded")
    except Exception as e:
        logger.warning(f"ML models not pre-loaded: {e}")

    logger.success(f"✅ API ready at http://{settings.api_host}:{settings.api_port}")

    yield  # App is running

    # ── Shutdown ─────────────────────────────────
    logger.info("Shutting down Smart Business AI Agent...")


# ─────────────────────────────────────────────
#  FastAPI App
# ─────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered business operations intelligence platform",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — allow Streamlit dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
#  Request Logging Middleware
# ─────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration_ms = int((time.time() - start_time) * 1000)

    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} ({duration_ms}ms)"
    )

    return response


# ─────────────────────────────────────────────
#  Exception Handlers
# ─────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "Contact support",
        },
    )


# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────

from backend.api.routes import agent, predict, anomaly, insights, data

app.include_router(agent.router, prefix=f"{settings.api_prefix}/agent", tags=["AI Agent"])
app.include_router(predict.router, prefix=f"{settings.api_prefix}/predict", tags=["Predictions"])
app.include_router(anomaly.router, prefix=f"{settings.api_prefix}/anomaly", tags=["Anomaly Detection"])
app.include_router(insights.router, prefix=f"{settings.api_prefix}/insights", tags=["Insights"])
app.include_router(data.router, prefix=f"{settings.api_prefix}/data", tags=["Data"])


# ─────────────────────────────────────────────
#  Health & Root
# ─────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["System"])
async def health_check():
    from backend.ml.model_manager import model_manager
    from backend.agents.business_agent import business_agent

    return {
        "status": "healthy",
        "timestamp": time.time(),
        "components": {
            "database": "connected",
            "ml_models": model_manager.status(),
            "llm": "available" if business_agent.is_llm_available else "offline (fallback active)",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
