"""
Anomaly Detection Model using Isolation Forest.

Isolation Forest is ideal for this use case because:
- Unsupervised: no labeled anomaly data needed
- Works on multivariate data (multiple metrics simultaneously)
- Fast and memory-efficient
- Produces anomaly scores (not just binary labels)

Features detected:
- Revenue anomalies (sudden drops/spikes)
- Margin compression
- Customer count anomalies
- Multi-variate combined anomalies
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class AnomalyDetectionModel:
    """
    Multivariate anomaly detection using Isolation Forest.

    Detects anomalies across multiple business metrics simultaneously,
    providing severity classification and contextual descriptions.
    """

    MODEL_FILENAME = "anomaly_detection.pkl"
    METADATA_FILENAME = "anomaly_metadata.pkl"

    SEVERITY_THRESHOLDS = {
        "critical": -0.5,   # Very strong anomaly
        "high": -0.3,        # Strong anomaly
        "medium": -0.1,      # Moderate anomaly
        "low": 0.0,          # Mild anomaly (score < 0 but close to boundary)
    }

    def __init__(self, models_dir: str = "./ml_models/trained"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline: Optional[Pipeline] = None
        self.is_trained = False
        self.metadata: Dict = {}
        self.feature_columns: List[str] = []

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for anomaly detection from raw sales data.
        Uses daily aggregated features with context.
        """
        df = df.copy()
        df = df.sort_values("date").reset_index(drop=True)

        # Daily aggregation
        daily = (
            df.groupby("date")
            .agg(
                total_revenue=("revenue", "sum"),
                avg_margin=("profit_margin", "mean"),
                total_units=("units_sold", "sum"),
                total_customers=("customer_count", "sum"),
                region_count=("region", "nunique"),
            )
            .reset_index()
            .sort_values("date")
            .reset_index(drop=True)
        )

        # Derived features
        daily["revenue_per_customer"] = (
            daily["total_revenue"] / daily["total_customers"].clip(lower=1)
        )
        daily["revenue_per_unit"] = (
            daily["total_revenue"] / daily["total_units"].clip(lower=1)
        )

        # Rolling z-scores (detect deviation from recent norm)
        for col in ["total_revenue", "avg_margin", "total_customers"]:
            rolling_mean = daily[col].rolling(30, min_periods=7).mean()
            rolling_std = daily[col].rolling(30, min_periods=7).std().clip(lower=1)
            daily[f"{col}_zscore"] = (daily[col] - rolling_mean) / rolling_std

        # Percentage change
        daily["revenue_pct_change"] = daily["total_revenue"].pct_change(periods=7).fillna(0)
        daily["margin_pct_change"] = daily["avg_margin"].pct_change(periods=7).fillna(0)

        # Drop rows with NaN (from rolling calculations)
        daily = daily.dropna().reset_index(drop=True)

        return daily

    def _get_feature_columns(self) -> List[str]:
        return [
            "total_revenue", "avg_margin", "total_units", "total_customers",
            "revenue_per_customer", "revenue_per_unit",
            "total_revenue_zscore", "avg_margin_zscore", "total_customers_zscore",
            "revenue_pct_change", "margin_pct_change",
        ]

    def train(self, df: pd.DataFrame) -> Dict:
        """Train Isolation Forest on historical data."""
        logger.info("Training anomaly detection model...")

        features_df = self._prepare_features(df)
        self.feature_columns = self._get_feature_columns()

        X = features_df[self.feature_columns].values

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", IsolationForest(
                n_estimators=200,
                contamination=0.05,     # Expect ~5% anomaly rate
                max_features=0.8,
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
            )),
        ])

        self.pipeline.fit(X)
        scores = self.pipeline.decision_function(X)
        labels = self.pipeline.predict(X)

        anomaly_count = int(np.sum(labels == -1))
        self.is_trained = True

        self.metadata = {
            "trained_at": datetime.utcnow().isoformat(),
            "training_samples": len(X),
            "anomaly_rate": round(anomaly_count / len(X), 4),
            "anomalies_found_in_training": anomaly_count,
            "feature_columns": self.feature_columns,
            "score_stats": {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
            }
        }

        logger.success(
            f"✅ Anomaly model trained | Samples: {len(X)} | "
            f"Anomaly rate: {anomaly_count/len(X):.1%}"
        )

        self.save()
        return self.metadata

    def detect(self, df: pd.DataFrame) -> List[Dict]:
        """
        Run anomaly detection on new data.

        Returns list of detected anomalies with severity and descriptions.
        """
        if not self.is_trained:
            self.load()

        features_df = self._prepare_features(df)
        X = features_df[self.feature_columns].values

        scores = self.pipeline.decision_function(X)
        labels = self.pipeline.predict(X)

        anomalies = []
        for idx, (score, label) in enumerate(zip(scores, labels)):
            if label == -1:  # Anomaly detected
                row = features_df.iloc[idx]
                severity = self._classify_severity(score)
                description = self._generate_description(row, score)

                anomalies.append({
                    "date": row["date"] if hasattr(row["date"], "isoformat") else row["date"],
                    "metric_name": "business_operations",
                    "metric_value": float(row["total_revenue"]),
                    "anomaly_score": round(float(score), 4),
                    "severity": severity,
                    "description": description,
                    "details": {
                        "revenue": round(float(row["total_revenue"]), 2),
                        "margin": round(float(row["avg_margin"]) * 100, 2),
                        "customers": int(row["total_customers"]),
                        "revenue_zscore": round(float(row["total_revenue_zscore"]), 2),
                    }
                })

        logger.info(f"Anomaly detection complete: {len(anomalies)} anomalies found")
        return sorted(anomalies, key=lambda x: x["anomaly_score"])

    def _classify_severity(self, score: float) -> str:
        """Convert anomaly score to severity label."""
        if score <= self.SEVERITY_THRESHOLDS["critical"]:
            return "critical"
        elif score <= self.SEVERITY_THRESHOLDS["high"]:
            return "high"
        elif score <= self.SEVERITY_THRESHOLDS["medium"]:
            return "medium"
        else:
            return "low"

    def _generate_description(self, row: pd.Series, score: float) -> str:
        """Generate human-readable anomaly description."""
        parts = []

        revenue_zscore = float(row.get("total_revenue_zscore", 0))
        margin_zscore = float(row.get("avg_margin_zscore", 0))
        rev_pct = float(row.get("revenue_pct_change", 0))

        if abs(revenue_zscore) > 2:
            direction = "spike" if revenue_zscore > 0 else "drop"
            parts.append(f"Revenue {direction} ({abs(revenue_zscore):.1f}σ from norm)")

        if abs(margin_zscore) > 2:
            direction = "increase" if margin_zscore > 0 else "compression"
            parts.append(f"Margin {direction} ({abs(margin_zscore):.1f}σ)")

        if abs(rev_pct) > 0.20:
            direction = "up" if rev_pct > 0 else "down"
            parts.append(f"Week-over-week revenue {direction} {abs(rev_pct):.0%}")

        if not parts:
            parts.append(f"Multi-variate anomaly (score: {score:.3f})")

        return "; ".join(parts)

    def save(self):
        joblib.dump(self.pipeline, self.models_dir / self.MODEL_FILENAME)
        joblib.dump(self.metadata, self.models_dir / self.METADATA_FILENAME)
        logger.info(f"Anomaly model saved to {self.models_dir}")

    def load(self):
        model_path = self.models_dir / self.MODEL_FILENAME
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run: python scripts/train_models.py"
            )
        self.pipeline = joblib.load(model_path)
        self.metadata = joblib.load(self.models_dir / self.METADATA_FILENAME)
        self.feature_columns = self.metadata["feature_columns"]
        self.is_trained = True

    def get_model_info(self) -> Dict:
        if not self.is_trained:
            try:
                self.load()
            except FileNotFoundError:
                return {"status": "not_trained"}
        return {"status": "trained", "model_type": "Isolation Forest", **self.metadata}
