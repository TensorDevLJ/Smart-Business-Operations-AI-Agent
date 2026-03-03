"""
Automated Insight Generation Engine.

Generates business insights by analyzing trends, patterns, and anomalies
in the data — no LLM required for basic insights (rule-based).
LLM can be used to summarize and make insights more natural.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np
from loguru import logger

from backend.database.connection import get_db
from backend.database.queries import (
    get_monthly_revenue,
    get_kpi_summary,
    get_revenue_by_region,
    get_revenue_by_category,
)


def calculate_trend(values: List[float]) -> Dict:
    """Calculate trend statistics for a series of values."""
    if len(values) < 2:
        return {"direction": "stable", "change_pct": 0.0, "slope": 0.0}

    x = np.arange(len(values))
    slope = np.polyfit(x, values, 1)[0]
    avg = np.mean(values)
    change_pct = (slope * len(values) / avg * 100) if avg != 0 else 0

    if change_pct > 5:
        direction = "growing"
    elif change_pct < -5:
        direction = "declining"
    else:
        direction = "stable"

    return {
        "direction": direction,
        "change_pct": round(change_pct, 2),
        "slope": round(float(slope), 2),
    }


def generate_revenue_insights(monthly_data: List[Dict]) -> List[Dict]:
    """Generate revenue trend insights."""
    insights = []

    if len(monthly_data) < 3:
        return insights

    revenues = [m["total_revenue"] for m in monthly_data]
    trend = calculate_trend(revenues)

    # Revenue trend insight
    insights.append({
        "type": "revenue_trend",
        "severity": "info" if trend["direction"] != "declining" else "warning",
        "title": f"Revenue is {trend['direction'].title()}",
        "description": (
            f"Revenue has {trend['direction']} by {abs(trend['change_pct']):.1f}% "
            f"over the last {len(monthly_data)} months."
        ),
        "metric": revenues[-1],
        "recommendation": (
            "Continue current growth strategy."
            if trend["direction"] == "growing"
            else "Investigate and address revenue decline immediately."
        ),
    })

    # Month-over-Month comparison
    if len(revenues) >= 2:
        mom_change = ((revenues[-1] - revenues[-2]) / revenues[-2] * 100) if revenues[-2] != 0 else 0
        severity = "success" if mom_change > 0 else ("warning" if mom_change < -5 else "info")

        insights.append({
            "type": "mom_change",
            "severity": severity,
            "title": f"Month-over-Month: {mom_change:+.1f}%",
            "description": (
                f"Revenue changed from ${revenues[-2]:,.0f} to ${revenues[-1]:,.0f} "
                f"({mom_change:+.1f}% change)."
            ),
            "metric": mom_change,
            "recommendation": (
                "Strong momentum — look to scale successful channels."
                if mom_change > 10
                else "Investigate root cause of revenue decline."
                if mom_change < -10
                else "Monitor closely for sustained trend."
            ),
        })

    # Best/worst performing months
    best_month = monthly_data[revenues.index(max(revenues))]
    worst_month = monthly_data[revenues.index(min(revenues))]

    insights.append({
        "type": "performance_extremes",
        "severity": "info",
        "title": "Performance Extremes Identified",
        "description": (
            f"Best month: {best_month['period']} (${best_month['total_revenue']:,.0f}). "
            f"Worst month: {worst_month['period']} (${worst_month['total_revenue']:,.0f})."
        ),
        "metric": max(revenues) - min(revenues),
        "recommendation": f"Study what drove success in {best_month['period']} and replicate.",
    })

    return insights


def generate_regional_insights(regional_data: List[Dict]) -> List[Dict]:
    """Generate regional performance insights."""
    insights = []

    if not regional_data:
        return insights

    total_rev = sum(r["total_revenue"] for r in regional_data)

    # Top region dominance
    if regional_data:
        top_region = regional_data[0]
        top_share = (top_region["total_revenue"] / total_rev * 100) if total_rev > 0 else 0

        insights.append({
            "type": "regional_concentration",
            "severity": "warning" if top_share > 60 else "info",
            "title": f"{top_region['region']} Dominates at {top_share:.0f}%",
            "description": (
                f"{top_region['region']} accounts for {top_share:.1f}% of total revenue "
                f"(${top_region['total_revenue']:,.0f})."
            ),
            "metric": top_share,
            "recommendation": (
                "High regional concentration — consider diversifying into underperforming regions."
                if top_share > 60
                else "Healthy regional distribution."
            ),
        })

    # Margin comparison across regions
    margins = [(r["region"], r["avg_margin_pct"]) for r in regional_data]
    if len(margins) >= 2:
        best_margin = max(margins, key=lambda x: x[1])
        worst_margin = min(margins, key=lambda x: x[1])

        insights.append({
            "type": "margin_disparity",
            "severity": "info",
            "title": "Margin Disparity Across Regions",
            "description": (
                f"Best margin: {best_margin[0]} ({best_margin[1]}%). "
                f"Lowest margin: {worst_margin[0]} ({worst_margin[1]}%)."
            ),
            "metric": best_margin[1] - worst_margin[1],
            "recommendation": (
                f"Investigate cost structure in {worst_margin[0]} to improve margins."
            ),
        })

    return insights


def generate_all_insights() -> Dict[str, Any]:
    """
    Master insight generator — combines all insight sources.
    Returns structured insights ready for API response.
    """
    try:
        with get_db() as db:
            monthly_data = get_monthly_revenue(db, months=12)
            kpis = get_kpi_summary(db)
            regional_data = get_revenue_by_region(db, months=3)
            category_data = get_revenue_by_category(db, months=3)

        all_insights = []
        all_insights.extend(generate_revenue_insights(monthly_data))
        all_insights.extend(generate_regional_insights(regional_data))

        # Critical alerts at top
        all_insights.sort(
            key=lambda x: {"critical": 0, "warning": 1, "info": 2, "success": 3}.get(x["severity"], 4)
        )

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "total_insights": len(all_insights),
            "kpi_snapshot": kpis,
            "insights": all_insights,
            "summary": f"Generated {len(all_insights)} business insights. "
                       f"Current MoM change: {kpis['mom_change_pct']:+.1f}%. "
                       f"Active anomalies: {kpis['active_anomalies']}.",
        }

    except Exception as e:
        logger.error(f"Insight generation error: {e}")
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "total_insights": 0,
            "insights": [],
            "error": str(e),
        }
