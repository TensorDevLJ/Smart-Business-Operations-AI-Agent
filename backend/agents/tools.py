"""
LangChain Tool definitions for the Business AI Agent.

Each tool wraps a specific backend capability:
- query_database: Get structured data from SQLite/PostgreSQL
- predict_sales: Call the forecasting ML model
- detect_anomalies: Run the Isolation Forest model
- get_kpi_metrics: Retrieve current KPIs
- get_insights: Generate automated business analysis

Design principles:
- Tools return plain strings (LangChain requirement)
- All errors are caught and returned as descriptive messages
- Each tool has clear docstrings for the LLM to understand when to use it
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
from datetime import datetime
from typing import Optional
from loguru import logger

from langchain.tools import tool

from backend.database.connection import get_db
from backend.database.queries import (
    get_monthly_revenue,
    get_revenue_by_region,
    get_revenue_by_category,
    get_kpi_summary,
    get_quarter_performance,
    get_sales_dataframe,
    get_recent_anomalies,
)
from backend.ml.model_manager import model_manager


@tool
def query_database(query_type: str) -> str:
    """
    Query the business database for historical data.

    Use this tool when the user asks about:
    - Historical sales/revenue data
    - Regional performance
    - Product category performance
    - Quarter or period-specific performance

    Args:
        query_type: One of: 'monthly_revenue', 'regional_performance',
                    'category_performance', 'kpi_summary',
                    'Q1_2024', 'Q2_2024', 'Q3_2024', 'Q4_2024',
                    'Q1_2025', 'Q2_2025', 'Q3_2025'
    """
    try:
        with get_db() as db:

            if query_type == "monthly_revenue":
                data = get_monthly_revenue(db, months=12)
                if not data:
                    return "No monthly revenue data found."
                summary = []
                for row in data[-6:]:  # Last 6 months
                    summary.append(
                        f"{row['period']}: ${row['total_revenue']:,.0f} revenue, "
                        f"{row['avg_margin']}% margin, "
                        f"{row['total_customers']:,} customers"
                    )
                return "Monthly Revenue (last 6 months):\n" + "\n".join(summary)

            elif query_type == "regional_performance":
                data = get_revenue_by_region(db, months=3)
                if not data:
                    return "No regional data found."
                summary = [f"Regional Performance (last 3 months):"]
                for row in data:
                    summary.append(
                        f"  {row['region']}: ${row['total_revenue']:,.0f} "
                        f"({row['avg_margin_pct']}% margin)"
                    )
                return "\n".join(summary)

            elif query_type == "category_performance":
                data = get_revenue_by_category(db, months=3)
                if not data:
                    return "No category data found."
                summary = ["Category Performance (last 3 months):"]
                for row in data:
                    summary.append(
                        f"  {row['category']}: ${row['total_revenue']:,.0f} "
                        f"({row['total_units']:,} units)"
                    )
                return "\n".join(summary)

            elif query_type == "kpi_summary":
                data = get_kpi_summary(db)
                return (
                    f"KPI Summary:\n"
                    f"  Current Month Revenue: ${data['current_month_revenue']:,.0f}\n"
                    f"  Last Month Revenue: ${data['last_month_revenue']:,.0f}\n"
                    f"  Month-over-Month Change: {data['mom_change_pct']:+.1f}%\n"
                    f"  Year-to-Date Revenue: ${data['ytd_revenue']:,.0f}\n"
                    f"  Active Anomalies: {data['active_anomalies']}"
                )

            elif query_type.startswith("Q"):
                # Parse Q1_2024, Q2_2025, etc.
                try:
                    parts = query_type.split("_")
                    quarter = int(parts[0][1])
                    year = int(parts[1])
                    data = get_quarter_performance(db, year, quarter)
                    return (
                        f"Quarter Performance — {data['period']}:\n"
                        f"  Revenue: ${data['revenue']:,.0f}\n"
                        f"  Units Sold: {data['units_sold']:,}\n"
                        f"  Avg Margin: {data['avg_margin_pct']}%\n"
                        f"  Customers: {data['customer_count']:,}"
                    )
                except (IndexError, ValueError) as e:
                    return f"Invalid quarter format '{query_type}'. Use 'Q1_2024' format."

            else:
                return (
                    f"Unknown query_type: '{query_type}'. "
                    "Valid options: monthly_revenue, regional_performance, "
                    "category_performance, kpi_summary, Q1_2024, Q2_2024, etc."
                )

    except Exception as e:
        logger.error(f"Database query error: {e}")
        return f"Database query failed: {str(e)}"


@tool
def predict_sales(months_ahead: int = 3) -> str:
    """
    Predict future sales/revenue using the ML forecasting model.

    Use this tool when the user asks about:
    - Future revenue predictions
    - Sales forecasts for upcoming months/quarters
    - "What will revenue be next quarter?"

    Args:
        months_ahead: Number of months to forecast (1-12, default 3)
    """
    try:
        months_ahead = max(1, min(12, int(months_ahead)))
        days_ahead = months_ahead * 30

        predictions = model_manager.forecasting.predict(periods=days_ahead)

        if not predictions:
            return "Unable to generate predictions. Ensure the model is trained."

        # Aggregate to monthly
        import pandas as pd
        df = pd.DataFrame(predictions)
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.to_period("M")

        monthly = df.groupby("month").agg(
            predicted_revenue=("predicted_revenue", "sum"),
            lower_bound=("lower_bound", "sum"),
            upper_bound=("upper_bound", "sum"),
        ).reset_index()

        model_info = model_manager.forecasting.metadata
        mape = model_info.get("cv_mape_mean", 0)

        result = [f"Sales Forecast (next {months_ahead} months) — Model accuracy: {(1-mape):.1%}:"]
        total_predicted = 0

        for _, row in monthly.iterrows():
            result.append(
                f"  {row['month']}: ${row['predicted_revenue']:,.0f} "
                f"(range: ${row['lower_bound']:,.0f} - ${row['upper_bound']:,.0f})"
            )
            total_predicted += row["predicted_revenue"]

        result.append(f"\nTotal {months_ahead}-month forecast: ${total_predicted:,.0f}")
        return "\n".join(result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return f"Prediction failed: {str(e)}. The model may need to be trained first."


@tool
def detect_anomalies(days_to_analyze: int = 30) -> str:
    """
    Detect anomalies in business operations using the Isolation Forest ML model.

    Use this tool when the user asks about:
    - Unusual patterns or outliers
    - "Why did sales drop in [period]?"
    - Operational issues or spikes
    - "Are there any anomalies?"

    Args:
        days_to_analyze: Number of recent days to analyze (default 30)
    """
    try:
        days_to_analyze = max(7, min(365, int(days_to_analyze)))

        with get_db() as db:
            df = get_sales_dataframe(db, months=max(3, days_to_analyze // 30 + 1))

        if df.empty:
            return "No data available for anomaly detection."

        anomalies = model_manager.anomaly.detect(df)

        if not anomalies:
            return f"✅ No anomalies detected in the last {days_to_analyze} days. Operations appear normal."

        # Filter to requested period
        from datetime import datetime, timedelta
        cutoff = datetime.utcnow() - timedelta(days=days_to_analyze)

        recent_anomalies = [
            a for a in anomalies
            if (isinstance(a["date"], str) and a["date"] >= cutoff.strftime("%Y-%m-%d"))
            or (hasattr(a["date"], "date") and a["date"] >= cutoff)
        ]

        if not recent_anomalies:
            return f"No anomalies detected in the last {days_to_analyze} days."

        severity_counts = {}
        for a in recent_anomalies:
            sev = a["severity"]
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        result = [
            f"⚠️ {len(recent_anomalies)} anomalies detected in last {days_to_analyze} days:",
            f"  Severity breakdown: {severity_counts}",
            "",
            "Top anomalies:",
        ]

        for a in recent_anomalies[:5]:  # Show top 5
            date_str = a["date"] if isinstance(a["date"], str) else a["date"].strftime("%Y-%m-%d")
            result.append(
                f"  [{a['severity'].upper()}] {date_str}: {a['description']} "
                f"(Revenue: ${a['metric_value']:,.0f})"
            )

        return "\n".join(result)

    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        return f"Anomaly detection failed: {str(e)}"


@tool
def get_kpi_metrics(metric_type: str = "all") -> str:
    """
    Retrieve current key performance indicators (KPIs).

    Use this tool for:
    - Current performance snapshots
    - Specific metric values
    - Dashboard summaries

    Args:
        metric_type: 'all', 'revenue', 'growth', 'margins'
    """
    try:
        with get_db() as db:
            kpis = get_kpi_summary(db)
            regional = get_revenue_by_region(db, months=1)
            categories = get_revenue_by_category(db, months=1)

        result = [
            "📊 Current KPI Summary:",
            f"  Current Month Revenue: ${kpis['current_month_revenue']:,.0f}",
            f"  MoM Change: {kpis['mom_change_pct']:+.1f}%",
            f"  YTD Revenue: ${kpis['ytd_revenue']:,.0f}",
            f"  Active Anomalies: {kpis['active_anomalies']}",
            "",
            "Top Regions (this month):",
        ]

        for r in regional[:3]:
            result.append(f"  {r['region']}: ${r['total_revenue']:,.0f}")

        result.append("\nTop Categories (this month):")
        for c in categories[:3]:
            result.append(f"  {c['category']}: ${c['total_revenue']:,.0f}")

        return "\n".join(result)

    except Exception as e:
        logger.error(f"KPI metrics error: {e}")
        return f"Failed to retrieve KPI metrics: {str(e)}"
