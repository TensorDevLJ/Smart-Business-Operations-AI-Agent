"""Data/Metrics API Routes."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from fastapi import APIRouter, Query
from loguru import logger

from backend.database.connection import get_db
from backend.database.queries import (
    get_monthly_revenue,
    get_revenue_by_region,
    get_revenue_by_category,
    get_kpi_summary,
)

router = APIRouter()


@router.get("/metrics")
async def get_metrics():
    """Get current KPI metrics."""
    with get_db() as db:
        return get_kpi_summary(db)


@router.get("/monthly-revenue")
async def get_monthly_revenue_data(months: int = Query(default=12, ge=1, le=36)):
    """Get monthly revenue time series."""
    with get_db() as db:
        data = get_monthly_revenue(db, months=months)
    return {"data": data, "months": months}


@router.get("/regional")
async def get_regional_data(months: int = Query(default=3, ge=1, le=24)):
    """Get regional revenue breakdown."""
    with get_db() as db:
        data = get_revenue_by_region(db, months=months)
    return {"data": data}


@router.get("/categories")
async def get_category_data(months: int = Query(default=3, ge=1, le=24)):
    """Get product category revenue breakdown."""
    with get_db() as db:
        data = get_revenue_by_category(db, months=months)
    return {"data": data}
