"""
Business query functions — all database interactions go through here.
Never write raw SQL in routes or agent tools; use these functions.
"""
from sqlalchemy.orm import Session
from sqlalchemy import func, extract, desc, and_
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd
from loguru import logger

from backend.database.models import SalesRecord, OperationalMetric, AnomalyLog, AlertLog


# ─────────────────────────────────────────────
#  SALES QUERIES
# ─────────────────────────────────────────────

def get_monthly_revenue(db: Session, months: int = 12) -> List[Dict]:
    """Get monthly revenue for the last N months."""
    cutoff = datetime.utcnow() - timedelta(days=months * 30)

    results = (
        db.query(
            extract("year", SalesRecord.date).label("year"),
            extract("month", SalesRecord.date).label("month"),
            func.sum(SalesRecord.revenue).label("total_revenue"),
            func.sum(SalesRecord.units_sold).label("total_units"),
            func.avg(SalesRecord.profit_margin).label("avg_margin"),
            func.sum(SalesRecord.customer_count).label("total_customers"),
        )
        .filter(SalesRecord.date >= cutoff)
        .group_by("year", "month")
        .order_by("year", "month")
        .all()
    )

    return [
        {
            "year": int(r.year),
            "month": int(r.month),
            "period": f"{int(r.year)}-{int(r.month):02d}",
            "total_revenue": round(r.total_revenue, 2),
            "total_units": int(r.total_units),
            "avg_margin": round(r.avg_margin * 100, 2),
            "total_customers": int(r.total_customers),
        }
        for r in results
    ]


def get_revenue_by_region(db: Session, months: int = 3) -> List[Dict]:
    """Get revenue breakdown by region."""
    cutoff = datetime.utcnow() - timedelta(days=months * 30)

    results = (
        db.query(
            SalesRecord.region,
            func.sum(SalesRecord.revenue).label("total_revenue"),
            func.avg(SalesRecord.profit_margin).label("avg_margin"),
        )
        .filter(SalesRecord.date >= cutoff)
        .group_by(SalesRecord.region)
        .order_by(desc("total_revenue"))
        .all()
    )

    return [
        {
            "region": r.region,
            "total_revenue": round(r.total_revenue, 2),
            "avg_margin_pct": round(r.avg_margin * 100, 2),
        }
        for r in results
    ]


def get_revenue_by_category(db: Session, months: int = 3) -> List[Dict]:
    """Get revenue breakdown by product category."""
    cutoff = datetime.utcnow() - timedelta(days=months * 30)

    results = (
        db.query(
            SalesRecord.product_category,
            func.sum(SalesRecord.revenue).label("total_revenue"),
            func.sum(SalesRecord.units_sold).label("total_units"),
        )
        .filter(SalesRecord.date >= cutoff)
        .group_by(SalesRecord.product_category)
        .order_by(desc("total_revenue"))
        .all()
    )

    return [
        {
            "category": r.product_category,
            "total_revenue": round(r.total_revenue, 2),
            "total_units": int(r.total_units),
        }
        for r in results
    ]


def get_kpi_summary(db: Session) -> Dict[str, Any]:
    """Get high-level KPI summary for dashboard header."""
    now = datetime.utcnow()
    current_month_start = now.replace(day=1, hour=0, minute=0, second=0)
    last_month_start = (current_month_start - timedelta(days=1)).replace(day=1)
    last_month_end = current_month_start - timedelta(seconds=1)

    def get_period_revenue(start, end):
        result = db.query(func.sum(SalesRecord.revenue)).filter(
            and_(SalesRecord.date >= start, SalesRecord.date <= end)
        ).scalar()
        return result or 0.0

    current_revenue = get_period_revenue(current_month_start, now)
    last_revenue = get_period_revenue(last_month_start, last_month_end)

    mom_change = 0.0
    if last_revenue > 0:
        mom_change = ((current_revenue - last_revenue) / last_revenue) * 100

    # YTD
    ytd_start = now.replace(month=1, day=1, hour=0, minute=0, second=0)
    ytd_revenue = get_period_revenue(ytd_start, now)

    # Total records
    total_records = db.query(func.count(SalesRecord.id)).scalar() or 0

    # Active anomalies
    active_anomalies = (
        db.query(func.count(AnomalyLog.id))
        .filter(AnomalyLog.is_resolved == False)
        .scalar()
        or 0
    )

    return {
        "current_month_revenue": round(current_revenue, 2),
        "last_month_revenue": round(last_revenue, 2),
        "mom_change_pct": round(mom_change, 2),
        "ytd_revenue": round(ytd_revenue, 2),
        "total_records": total_records,
        "active_anomalies": active_anomalies,
    }


def get_sales_dataframe(db: Session, months: int = 24) -> pd.DataFrame:
    """Get sales data as DataFrame for ML model input."""
    cutoff = datetime.utcnow() - timedelta(days=months * 30)

    records = (
        db.query(SalesRecord)
        .filter(SalesRecord.date >= cutoff)
        .order_by(SalesRecord.date)
        .all()
    )

    if not records:
        return pd.DataFrame()

    data = [
        {
            "date": r.date,
            "revenue": r.revenue,
            "units_sold": r.units_sold,
            "profit_margin": r.profit_margin,
            "customer_count": r.customer_count,
            "region": r.region,
            "product_category": r.product_category,
            "channel": r.channel,
        }
        for r in records
    ]

    return pd.DataFrame(data)


def get_quarter_performance(db: Session, year: int, quarter: int) -> Dict:
    """Get performance for a specific quarter."""
    quarter_months = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}
    start_month, end_month = quarter_months[quarter]

    start = datetime(year, start_month, 1)
    if end_month == 12:
        end = datetime(year + 1, 1, 1) - timedelta(seconds=1)
    else:
        end = datetime(year, end_month + 1, 1) - timedelta(seconds=1)

    result = db.query(
        func.sum(SalesRecord.revenue).label("revenue"),
        func.sum(SalesRecord.units_sold).label("units"),
        func.avg(SalesRecord.profit_margin).label("margin"),
        func.sum(SalesRecord.customer_count).label("customers"),
    ).filter(and_(SalesRecord.date >= start, SalesRecord.date <= end)).first()

    return {
        "year": year,
        "quarter": quarter,
        "period": f"Q{quarter} {year}",
        "revenue": round(result.revenue or 0, 2),
        "units_sold": int(result.units or 0),
        "avg_margin_pct": round((result.margin or 0) * 100, 2),
        "customer_count": int(result.customers or 0),
    }


# ─────────────────────────────────────────────
#  ANOMALY QUERIES
# ─────────────────────────────────────────────

def save_anomalies(db: Session, anomalies: List[Dict]) -> int:
    """Persist detected anomalies to database."""
    saved = 0
    for a in anomalies:
        log = AnomalyLog(
            record_date=a["date"],
            metric_name=a["metric_name"],
            metric_value=a["metric_value"],
            anomaly_score=a["anomaly_score"],
            severity=a["severity"],
            description=a.get("description", ""),
        )
        db.add(log)
        saved += 1
    db.commit()
    return saved


def get_recent_anomalies(db: Session, days: int = 30) -> List[Dict]:
    """Get anomalies from the last N days."""
    cutoff = datetime.utcnow() - timedelta(days=days)

    anomalies = (
        db.query(AnomalyLog)
        .filter(AnomalyLog.detected_at >= cutoff)
        .order_by(desc(AnomalyLog.detected_at))
        .all()
    )

    return [
        {
            "id": a.id,
            "detected_at": a.detected_at.isoformat(),
            "record_date": a.record_date.isoformat(),
            "metric_name": a.metric_name,
            "metric_value": a.metric_value,
            "severity": a.severity,
            "description": a.description,
            "is_resolved": a.is_resolved,
        }
        for a in anomalies
    ]
