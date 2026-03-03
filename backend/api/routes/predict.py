"""Sales Prediction API Routes."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict
from loguru import logger

from backend.ml.model_manager import model_manager

router = APIRouter()


@router.post("/sales")
async def predict_sales(months_ahead: int = Query(default=3, ge=1, le=12)):
    """
    Predict future sales for N months ahead.
    Returns daily predictions aggregated to monthly view.
    """
    try:
        days = months_ahead * 30
        predictions = model_manager.forecasting.predict(periods=days)
        model_info = model_manager.forecasting.get_model_info()

        # Aggregate to monthly
        from collections import defaultdict
        monthly = defaultdict(lambda: {"revenue": 0.0, "lower": 0.0, "upper": 0.0, "days": 0})

        for p in predictions:
            month_key = p["date"][:7]  # YYYY-MM
            monthly[month_key]["revenue"] += p["predicted_revenue"]
            monthly[month_key]["lower"] += p["lower_bound"]
            monthly[month_key]["upper"] += p["upper_bound"]
            monthly[month_key]["days"] += 1

        monthly_list = [
            {
                "month": k,
                "predicted_revenue": round(v["revenue"], 2),
                "lower_bound": round(v["lower"], 2),
                "upper_bound": round(v["upper"], 2),
            }
            for k, v in sorted(monthly.items())
        ]

        return {
            "predictions": monthly_list,
            "daily_predictions": predictions[:30],  # First month detail
            "model_accuracy": {
                "mape": model_info.get("cv_mape_mean"),
                "r2": model_info.get("final_r2"),
            },
            "total_predicted_revenue": round(sum(v["revenue"] for v in monthly.values()), 2),
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Run: python scripts/train_models.py"
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-info")
async def get_model_info():
    """Get forecasting model metadata."""
    return model_manager.forecasting.get_model_info()
