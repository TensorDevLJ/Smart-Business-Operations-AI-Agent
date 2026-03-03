"""Anomaly Detection API Routes."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from backend.ml.model_manager import model_manager
from backend.database.connection import get_db
from backend.database.queries import get_sales_dataframe, get_recent_anomalies, save_anomalies

router = APIRouter()


@router.get("/detect")
async def detect_anomalies(days: int = Query(default=90, ge=7, le=365)):
    """
    Run anomaly detection on recent business operations data.
    Persists detected anomalies to database for audit trail.
    """
    try:
        with get_db() as db:
            df = get_sales_dataframe(db, months=max(3, days // 30 + 1))

        if df.empty:
            return {"anomalies": [], "message": "No data available for analysis"}

        anomalies = model_manager.anomaly.detect(df)

        # Persist to database
        if anomalies:
            with get_db() as db:
                saved = save_anomalies(db, anomalies)
                logger.info(f"Saved {saved} anomalies to database")

        severity_counts = {}
        for a in anomalies:
            sev = a["severity"]
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        return {
            "total_anomalies": len(anomalies),
            "severity_breakdown": severity_counts,
            "anomalies": [
                {
                    "date": str(a["date"]),
                    "severity": a["severity"],
                    "description": a["description"],
                    "revenue": a.get("details", {}).get("revenue"),
                    "anomaly_score": a["anomaly_score"],
                }
                for a in anomalies[:20]  # Top 20
            ],
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Anomaly model not trained. Run: python scripts/train_models.py"
        )
    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_anomaly_history(days: int = Query(default=30, ge=1, le=365)):
    """Get historical anomalies from database."""
    with get_db() as db:
        anomalies = get_recent_anomalies(db, days=days)
    return {"anomalies": anomalies, "count": len(anomalies)}
