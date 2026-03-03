"""Business Insights API Routes."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from fastapi import APIRouter
from loguru import logger

from backend.utils.insights import generate_all_insights
from backend.utils.alerts import run_all_alert_checks, get_recent_alerts

router = APIRouter()


@router.get("/summary")
async def get_insights_summary():
    """Auto-generate business insights from current data."""
    return generate_all_insights()


@router.get("/alerts")
async def get_alerts(days: int = 7):
    """Get recent alerts."""
    return {"alerts": get_recent_alerts(days=days)}


@router.post("/alerts/check")
async def trigger_alert_check():
    """Manually trigger alert checks."""
    result = run_all_alert_checks()
    return result
