"""
Automated Alert System.

Checks business metrics against thresholds and triggers alerts.
Alerts are:
1. Stored in the database (AlertLog)
2. Logged for monitoring
3. Can trigger email/webhook (if configured)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from datetime import datetime, timedelta
from typing import List, Dict
from loguru import logger

from backend.database.connection import get_db
from backend.database.models import AlertLog
from backend.database.queries import get_kpi_summary, get_monthly_revenue
from config.settings import settings


def check_revenue_drop_alert(db) -> List[Dict]:
    """Alert if revenue drops more than threshold vs previous period."""
    alerts = []
    kpis = get_kpi_summary(db)

    mom_change = kpis["mom_change_pct"] / 100  # Convert to decimal
    threshold = settings.alert_revenue_drop_threshold

    if mom_change < -threshold:
        severity = "critical" if mom_change < -0.25 else "high"
        alert = {
            "alert_type": "revenue_drop",
            "severity": severity,
            "message": (
                f"Revenue dropped {abs(mom_change):.1%} month-over-month. "
                f"Current: ${kpis['current_month_revenue']:,.0f}, "
                f"Previous: ${kpis['last_month_revenue']:,.0f}. "
                f"Threshold: {threshold:.0%}"
            ),
            "metric_value": kpis["current_month_revenue"],
            "threshold_value": kpis["last_month_revenue"] * (1 - threshold),
        }
        alerts.append(alert)
        logger.warning(f"ALERT [{severity.upper()}]: Revenue drop detected: {abs(mom_change):.1%}")

    return alerts


def check_anomaly_count_alert(db) -> List[Dict]:
    """Alert if too many active anomalies."""
    alerts = []
    from sqlalchemy import func
    from backend.database.models import AnomalyLog

    active_count = (
        db.query(func.count(AnomalyLog.id))
        .filter(AnomalyLog.is_resolved == False)
        .scalar() or 0
    )

    if active_count >= 10:
        alerts.append({
            "alert_type": "anomaly_surge",
            "severity": "critical" if active_count >= 20 else "high",
            "message": f"{active_count} unresolved anomalies detected. Immediate investigation required.",
            "metric_value": float(active_count),
            "threshold_value": 10.0,
        })

    return alerts


def run_all_alert_checks() -> Dict:
    """
    Run all alert checks and persist triggered alerts to DB.
    Called periodically by the scheduler.
    """
    all_alerts = []

    try:
        with get_db() as db:
            all_alerts.extend(check_revenue_drop_alert(db))
            all_alerts.extend(check_anomaly_count_alert(db))

            # Persist new alerts
            for alert_data in all_alerts:
                alert = AlertLog(**alert_data)
                db.add(alert)

            db.commit()

    except Exception as e:
        logger.error(f"Alert check error: {e}")

    logger.info(f"Alert check complete: {len(all_alerts)} alerts triggered")

    return {
        "checked_at": datetime.utcnow().isoformat(),
        "alerts_triggered": len(all_alerts),
        "alerts": all_alerts,
    }


def get_recent_alerts(days: int = 7) -> List[Dict]:
    """Get recent alerts for dashboard display."""
    cutoff = datetime.utcnow() - timedelta(days=days)

    with get_db() as db:
        from sqlalchemy import desc
        alerts = (
            db.query(AlertLog)
            .filter(AlertLog.triggered_at >= cutoff)
            .order_by(desc(AlertLog.triggered_at))
            .limit(50)
            .all()
        )

    return [
        {
            "id": a.id,
            "triggered_at": a.triggered_at.isoformat(),
            "alert_type": a.alert_type,
            "severity": a.severity,
            "message": a.message,
            "metric_value": a.metric_value,
            "is_acknowledged": a.is_acknowledged,
        }
        for a in alerts
    ]
