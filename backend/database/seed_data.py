"""
Generates realistic synthetic business data for development and demos.
Run: python scripts/seed_database.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

from backend.database.connection import init_db, get_db
from backend.database.models import SalesRecord, OperationalMetric


REGIONS = ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East"]
CATEGORIES = ["Software", "Hardware", "Services", "Consulting", "Support"]
CHANNELS = ["online", "retail", "wholesale", "direct"]

# Seasonal factors by month (1=Jan, 12=Dec)
SEASONAL_FACTORS = {
    1: 0.75, 2: 0.80, 3: 0.90, 4: 0.95, 5: 1.00, 6: 1.05,
    7: 0.95, 8: 0.90, 9: 1.05, 10: 1.10, 11: 1.20, 12: 1.35,
}

# Region revenue weight
REGION_WEIGHTS = {
    "North America": 0.40,
    "Europe": 0.28,
    "Asia Pacific": 0.18,
    "Latin America": 0.09,
    "Middle East": 0.05,
}

# Category margin profiles
CATEGORY_MARGINS = {
    "Software": (0.70, 0.85),
    "Hardware": (0.25, 0.40),
    "Services": (0.55, 0.70),
    "Consulting": (0.60, 0.75),
    "Support": (0.50, 0.65),
}


def generate_sales_data(
    start_date: datetime,
    end_date: datetime,
    base_daily_revenue: float = 150_000,
    inject_anomalies: bool = True,
) -> list:
    """
    Generate synthetic daily sales records with:
    - Seasonal patterns
    - Weekly patterns (weekends lower)
    - Trend growth (5% YoY)
    - Injected anomalies for testing
    """
    records = []
    current = start_date
    day_idx = 0

    # Anomaly dates (inject low/high spikes for March drop narrative)
    anomaly_drops = {
        datetime(2024, 3, 15): 0.35,  # "March sales drop" scenario
        datetime(2024, 3, 16): 0.40,
        datetime(2024, 3, 17): 0.45,
        datetime(2024, 8, 10): 0.30,  # Summer anomaly
    }
    anomaly_spikes = {
        datetime(2024, 11, 29): 3.5,  # Black Friday spike
        datetime(2024, 12, 1): 2.8,   # Cyber Monday
    }

    while current <= end_date:
        seasonal = SEASONAL_FACTORS.get(current.month, 1.0)
        weekday_factor = 0.65 if current.weekday() >= 5 else 1.0  # Weekend dip
        trend_factor = 1 + (day_idx / 365) * 0.05  # 5% annual growth
        noise = np.random.normal(1.0, 0.08)

        daily_rev = base_daily_revenue * seasonal * weekday_factor * trend_factor * noise

        # Apply anomalies
        anomaly_date_key = current.replace(hour=0, minute=0, second=0, microsecond=0)
        if inject_anomalies and anomaly_date_key in anomaly_drops:
            daily_rev *= anomaly_drops[anomaly_date_key]
        if inject_anomalies and anomaly_date_key in anomaly_spikes:
            daily_rev *= anomaly_spikes[anomaly_date_key]

        # Distribute revenue across regions and categories
        for region in REGIONS:
            region_rev = daily_rev * REGION_WEIGHTS[region] * np.random.uniform(0.9, 1.1)

            for category in CATEGORIES:
                cat_share = np.random.dirichlet(np.ones(len(CATEGORIES)))[CATEGORIES.index(category)]
                cat_rev = region_rev * cat_share

                if cat_rev < 100:  # Skip negligible records
                    continue

                margin_low, margin_high = CATEGORY_MARGINS[category]
                margin = np.random.uniform(margin_low, margin_high)

                units = max(1, int(cat_rev / np.random.uniform(50, 500)))
                customers = max(1, int(units * np.random.uniform(0.3, 0.8)))
                channel = np.random.choice(CHANNELS, p=[0.45, 0.25, 0.20, 0.10])

                records.append({
                    "date": current,
                    "region": region,
                    "product_category": category,
                    "revenue": round(cat_rev, 2),
                    "units_sold": units,
                    "profit_margin": round(margin, 4),
                    "customer_count": customers,
                    "channel": channel,
                })

        current += timedelta(days=1)
        day_idx += 1

    return records


def generate_operational_metrics(
    start_date: datetime,
    end_date: datetime,
) -> list:
    """Generate operational KPI metrics."""
    metrics = []
    current = start_date

    base_metrics = {
        "customer_satisfaction": (4.2, 0.15),   # mean, std
        "support_tickets": (450, 60),
        "avg_response_time_hrs": (2.5, 0.8),
        "employee_productivity": (85, 8),
        "inventory_turnover": (6.2, 0.9),
        "churn_rate": (0.023, 0.008),
    }

    while current <= end_date:
        for metric_name, (mean, std) in base_metrics.items():
            value = np.random.normal(mean, std)

            unit_map = {
                "customer_satisfaction": "score",
                "support_tickets": "count",
                "avg_response_time_hrs": "hours",
                "employee_productivity": "percent",
                "inventory_turnover": "ratio",
                "churn_rate": "percent",
            }

            metrics.append({
                "date": current,
                "metric_name": metric_name,
                "metric_value": round(float(value), 4),
                "unit": unit_map[metric_name],
                "department": "operations",
            })

        current += timedelta(days=1)

    return metrics


def seed_database(records_per_batch: int = 500):
    """Main seeding function."""
    logger.info("Starting database seeding...")
    init_db()

    start_date = datetime(2023, 1, 1)
    end_date = datetime.utcnow()

    # ── Sales Data ──────────────────────────────────
    logger.info("Generating sales records...")
    sales_data = generate_sales_data(start_date, end_date)
    logger.info(f"Generated {len(sales_data)} sales records")

    with get_db() as db:
        # Clear existing
        db.query(SalesRecord).delete()

        # Batch insert
        batch = []
        for i, row in enumerate(sales_data):
            batch.append(SalesRecord(**row))
            if len(batch) >= records_per_batch:
                db.bulk_save_objects(batch)
                db.flush()
                batch = []
                logger.info(f"  Inserted {i + 1}/{len(sales_data)} sales records...")

        if batch:
            db.bulk_save_objects(batch)

        db.commit()
        logger.success(f"✅ Inserted {len(sales_data)} sales records")

    # ── Operational Metrics ──────────────────────────
    logger.info("Generating operational metrics...")
    ops_data = generate_operational_metrics(start_date, end_date)

    with get_db() as db:
        db.query(OperationalMetric).delete()

        batch = []
        for i, row in enumerate(ops_data):
            batch.append(OperationalMetric(**row))
            if len(batch) >= records_per_batch:
                db.bulk_save_objects(batch)
                db.flush()
                batch = []

        if batch:
            db.bulk_save_objects(batch)

        db.commit()
        logger.success(f"✅ Inserted {len(ops_data)} operational metric records")

    logger.success("🎉 Database seeding complete!")


if __name__ == "__main__":
    seed_database()
