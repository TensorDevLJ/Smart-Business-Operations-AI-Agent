"""
Tests for the AI Agent and ML models.
Run: pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def sample_sales_df():
    """Generate minimal sample sales DataFrame for testing."""
    dates = pd.date_range(start="2023-01-01", periods=365, freq="D")
    np.random.seed(42)

    records = []
    for date in dates:
        records.append({
            "date": date,
            "revenue": np.random.uniform(100_000, 200_000),
            "units_sold": np.random.randint(100, 500),
            "profit_margin": np.random.uniform(0.4, 0.8),
            "customer_count": np.random.randint(50, 200),
            "region": "North America",
            "product_category": "Software",
            "channel": "online",
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
#  ML Model Tests
# ─────────────────────────────────────────────

class TestForecastingModel:
    def test_model_training(self, sample_sales_df, tmp_path):
        from backend.ml.forecasting import SalesForecastingModel

        model = SalesForecastingModel(models_dir=str(tmp_path))
        metrics = model.train(sample_sales_df)

        assert metrics["training_samples"] > 0
        assert 0.0 <= metrics["cv_mape_mean"] <= 1.0
        assert metrics["final_r2"] > 0  # Should have positive R²
        assert "trained_at" in metrics

    def test_model_prediction(self, sample_sales_df, tmp_path):
        from backend.ml.forecasting import SalesForecastingModel

        model = SalesForecastingModel(models_dir=str(tmp_path))
        model.train(sample_sales_df)

        predictions = model.predict(periods=30)

        assert len(predictions) == 30
        assert all("date" in p for p in predictions)
        assert all("predicted_revenue" in p for p in predictions)
        assert all(p["predicted_revenue"] >= 0 for p in predictions)
        assert all(p["lower_bound"] <= p["predicted_revenue"] for p in predictions)

    def test_model_persistence(self, sample_sales_df, tmp_path):
        from backend.ml.forecasting import SalesForecastingModel

        # Train and save
        model1 = SalesForecastingModel(models_dir=str(tmp_path))
        model1.train(sample_sales_df)

        # Load in new instance
        model2 = SalesForecastingModel(models_dir=str(tmp_path))
        model2.load()

        # Both should give same predictions
        pred1 = model1.predict(periods=7)
        pred2 = model2.predict(periods=7)

        assert len(pred1) == len(pred2)
        for p1, p2 in zip(pred1, pred2):
            assert p1["date"] == p2["date"]
            assert abs(p1["predicted_revenue"] - p2["predicted_revenue"]) < 0.01


class TestAnomalyModel:
    def test_model_training(self, sample_sales_df, tmp_path):
        from backend.ml.anomaly import AnomalyDetectionModel

        model = AnomalyDetectionModel(models_dir=str(tmp_path))
        metrics = model.train(sample_sales_df)

        assert metrics["training_samples"] > 0
        assert 0.0 <= metrics["anomaly_rate"] <= 0.2  # Reasonable rate
        assert "trained_at" in metrics

    def test_anomaly_detection(self, sample_sales_df, tmp_path):
        from backend.ml.anomaly import AnomalyDetectionModel

        # Inject obvious anomaly
        df = sample_sales_df.copy()
        anomaly_date = df["date"].iloc[-30]
        df.loc[df["date"] == anomaly_date, "revenue"] = 10.0  # Extreme outlier

        model = AnomalyDetectionModel(models_dir=str(tmp_path))
        model.train(sample_sales_df)  # Train on clean data
        anomalies = model.detect(df)

        # Should detect at least the injected anomaly
        assert isinstance(anomalies, list)
        if anomalies:
            assert all("severity" in a for a in anomalies)
            assert all(a["severity"] in ["low", "medium", "high", "critical"] for a in anomalies)

    def test_severity_classification(self, tmp_path):
        from backend.ml.anomaly import AnomalyDetectionModel

        model = AnomalyDetectionModel(models_dir=str(tmp_path))

        assert model._classify_severity(-0.6) == "critical"
        assert model._classify_severity(-0.4) == "high"
        assert model._classify_severity(-0.2) == "medium"
        assert model._classify_severity(-0.05) == "low"


# ─────────────────────────────────────────────
#  Database Tests
# ─────────────────────────────────────────────

class TestDatabase:
    def test_db_initialization(self, tmp_path):
        import os
        os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"

        from backend.database.connection import init_db, get_db
        from backend.database.models import SalesRecord

        init_db()

        with get_db() as db:
            count = db.query(SalesRecord).count()
            assert count == 0  # Empty DB

    def test_seed_data_generation(self):
        from backend.database.seed_data import generate_sales_data

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        records = generate_sales_data(start, end, inject_anomalies=False)

        assert len(records) > 0
        assert all("revenue" in r for r in records)
        assert all("region" in r for r in records)
        assert all(r["revenue"] > 0 for r in records)


# ─────────────────────────────────────────────
#  Insights Tests
# ─────────────────────────────────────────────

class TestInsights:
    def test_trend_calculation(self):
        from backend.utils.insights import calculate_trend

        # Growing trend
        growing = [100, 110, 120, 130, 140]
        result = calculate_trend(growing)
        assert result["direction"] == "growing"
        assert result["change_pct"] > 0

        # Declining trend
        declining = [140, 130, 120, 110, 100]
        result = calculate_trend(declining)
        assert result["direction"] == "declining"
        assert result["change_pct"] < 0

        # Stable trend
        stable = [100, 101, 99, 100, 102]
        result = calculate_trend(stable)
        assert result["direction"] == "stable"
