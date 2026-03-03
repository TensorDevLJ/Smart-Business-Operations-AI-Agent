"""
Sales Forecasting ML Model.

Uses a combination of:
- Linear Regression with polynomial features for trend capture
- Feature engineering: lag features, rolling stats, seasonality encoding
- Model persistence with joblib

Run training via: python scripts/train_models.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from loguru import logger
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


class SalesForecastingModel:
    """
    Production-grade sales forecasting model with feature engineering.

    Architecture:
    - Features: time index, month, quarter, day-of-week, lag features, rolling means
    - Model: Ridge regression with polynomial features (degree=2)
    - Validation: TimeSeriesSplit (respects temporal order)
    """

    MODEL_FILENAME = "sales_forecasting.pkl"
    SCALER_FILENAME = "sales_scaler.pkl"
    METADATA_FILENAME = "sales_model_metadata.pkl"

    def __init__(self, models_dir: str = "./ml_models/trained"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline: Optional[Pipeline] = None
        self.is_trained = False
        self.metadata: Dict = {}
        self.feature_columns: List[str] = []

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw time-series data into ML features.

        Features:
        - Time: ordinal index, month, quarter, day_of_week, is_weekend
        - Lags: revenue_lag_1, lag_7, lag_30 (previous periods)
        - Rolling: rolling_mean_7, rolling_mean_30, rolling_std_7
        - Cyclical encoding: sin/cos for month (captures periodicity)
        """
        df = df.copy()
        df = df.sort_values("date").reset_index(drop=True)

        # Aggregate to daily if needed
        if "revenue" in df.columns:
            daily = (
                df.groupby("date")["revenue"]
                .sum()
                .reset_index()
                .sort_values("date")
                .reset_index(drop=True)
            )
        else:
            daily = df[["date", "revenue"]].copy()

        # Time features
        daily["time_idx"] = range(len(daily))
        daily["month"] = daily["date"].dt.month
        daily["quarter"] = daily["date"].dt.quarter
        daily["day_of_week"] = daily["date"].dt.dayofweek
        daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)
        daily["day_of_year"] = daily["date"].dt.dayofyear
        daily["year"] = daily["date"].dt.year

        # Cyclical encoding (prevents discontinuity at Dec→Jan)
        daily["month_sin"] = np.sin(2 * np.pi * daily["month"] / 12)
        daily["month_cos"] = np.cos(2 * np.pi * daily["month"] / 12)
        daily["dow_sin"] = np.sin(2 * np.pi * daily["day_of_week"] / 7)
        daily["dow_cos"] = np.cos(2 * np.pi * daily["day_of_week"] / 7)

        # Lag features
        for lag in [1, 7, 14, 30, 90]:
            daily[f"revenue_lag_{lag}"] = daily["revenue"].shift(lag)

        # Rolling statistics
        for window in [7, 14, 30]:
            daily[f"rolling_mean_{window}"] = (
                daily["revenue"].shift(1).rolling(window).mean()
            )
            daily[f"rolling_std_{window}"] = (
                daily["revenue"].shift(1).rolling(window).std()
            )

        # Drop NaN rows from lag/rolling (lose first 90 rows)
        daily = daily.dropna().reset_index(drop=True)

        return daily

    def _get_feature_columns(self) -> List[str]:
        """List of feature column names used in training."""
        cols = [
            "time_idx", "month", "quarter", "day_of_week", "is_weekend",
            "day_of_year", "year",
            "month_sin", "month_cos", "dow_sin", "dow_cos",
        ]
        for lag in [1, 7, 14, 30, 90]:
            cols.append(f"revenue_lag_{lag}")
        for window in [7, 14, 30]:
            cols.append(f"rolling_mean_{window}")
            cols.append(f"rolling_std_{window}")
        return cols

    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train the forecasting model.

        Args:
            df: DataFrame with columns [date, revenue, ...]

        Returns:
            Training metrics dictionary
        """
        logger.info("Starting sales forecasting model training...")

        engineered = self._engineer_features(df)
        self.feature_columns = self._get_feature_columns()

        X = engineered[self.feature_columns].values
        y = engineered["revenue"].values

        if len(X) < 100:
            raise ValueError(f"Insufficient training data: {len(X)} samples (need >= 100)")

        # Time-series cross-validation (never leak future into past)
        tscv = TimeSeriesSplit(n_splits=5)

        # Pipeline: Scaler → Polynomial → Ridge
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", Ridge(alpha=10.0)),
        ])

        # Evaluate with CV
        cv_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            self.pipeline.fit(X_train, y_train)
            y_pred = self.pipeline.predict(X_val)
            mape = mean_absolute_percentage_error(y_val, y_pred)
            cv_scores.append(mape)

        # Final training on all data
        self.pipeline.fit(X, y)
        y_pred_all = self.pipeline.predict(X)

        final_mape = mean_absolute_percentage_error(y, y_pred_all)
        final_r2 = r2_score(y, y_pred_all)

        self.is_trained = True
        self.metadata = {
            "trained_at": datetime.utcnow().isoformat(),
            "training_samples": len(X),
            "cv_mape_mean": round(float(np.mean(cv_scores)), 4),
            "cv_mape_std": round(float(np.std(cv_scores)), 4),
            "final_mape": round(final_mape, 4),
            "final_r2": round(final_r2, 4),
            "feature_count": len(self.feature_columns),
            "date_range": {
                "start": engineered["date"].min().isoformat(),
                "end": engineered["date"].max().isoformat(),
            },
            # Store last 90 days of data for prediction continuation
            "last_values": engineered["revenue"].tail(90).tolist(),
            "last_date": engineered["date"].max(),
            "feature_columns": self.feature_columns,
        }

        logger.success(
            f"✅ Model trained | MAPE: {final_mape:.2%} | R²: {final_r2:.4f} | "
            f"CV MAPE: {np.mean(cv_scores):.2%} ± {np.std(cv_scores):.2%}"
        )

        self.save()
        return self.metadata

    def predict(self, periods: int = 90) -> List[Dict]:
        """
        Generate future predictions for N days ahead.

        Returns list of {date, predicted_revenue, lower_bound, upper_bound}
        """
        if not self.is_trained:
            self.load()

        last_date = self.metadata["last_date"]
        if isinstance(last_date, str):
            last_date = datetime.fromisoformat(last_date)

        # Reconstruct the historical series for lag features
        historical = self.metadata["last_values"].copy()
        predictions = []

        for i in range(1, periods + 1):
            pred_date = last_date + timedelta(days=i)
            time_idx = self.metadata["training_samples"] + i

            # Build single row of features
            row = {
                "time_idx": time_idx,
                "month": pred_date.month,
                "quarter": (pred_date.month - 1) // 3 + 1,
                "day_of_week": pred_date.weekday(),
                "is_weekend": int(pred_date.weekday() >= 5),
                "day_of_year": pred_date.timetuple().tm_yday,
                "year": pred_date.year,
                "month_sin": np.sin(2 * np.pi * pred_date.month / 12),
                "month_cos": np.cos(2 * np.pi * pred_date.month / 12),
                "dow_sin": np.sin(2 * np.pi * pred_date.weekday() / 7),
                "dow_cos": np.cos(2 * np.pi * pred_date.weekday() / 7),
            }

            # Lag features from combined historical + predicted values
            all_values = historical + [p["predicted_revenue"] for p in predictions]
            for lag in [1, 7, 14, 30, 90]:
                idx = -(lag)
                row[f"revenue_lag_{lag}"] = all_values[idx] if len(all_values) >= lag else np.mean(historical[-30:])

            # Rolling stats
            for window in [7, 14, 30]:
                recent = all_values[-window:] if len(all_values) >= window else historical[-window:]
                row[f"rolling_mean_{window}"] = float(np.mean(recent))
                row[f"rolling_std_{window}"] = float(np.std(recent)) if len(recent) > 1 else 0.0

            X = np.array([[row[col] for col in self.feature_columns]])
            pred_value = float(self.pipeline.predict(X)[0])

            # Confidence interval (approximate from CV error)
            cv_mape = self.metadata["cv_mape_mean"]
            margin = pred_value * cv_mape * 1.5

            predictions.append({
                "date": pred_date.strftime("%Y-%m-%d"),
                "predicted_revenue": round(max(0, pred_value), 2),
                "lower_bound": round(max(0, pred_value - margin), 2),
                "upper_bound": round(pred_value + margin, 2),
            })

        return predictions

    def save(self):
        """Persist model and metadata to disk."""
        joblib.dump(self.pipeline, self.models_dir / self.MODEL_FILENAME)
        joblib.dump(self.metadata, self.models_dir / self.METADATA_FILENAME)
        logger.info(f"Model saved to {self.models_dir}")

    def load(self):
        """Load model from disk."""
        model_path = self.models_dir / self.MODEL_FILENAME
        meta_path = self.models_dir / self.METADATA_FILENAME

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run: python scripts/train_models.py"
            )

        self.pipeline = joblib.load(model_path)
        self.metadata = joblib.load(meta_path)
        self.feature_columns = self.metadata["feature_columns"]
        self.is_trained = True
        logger.info(f"Model loaded from {self.models_dir}")

    def get_model_info(self) -> Dict:
        """Return model metadata for API responses."""
        if not self.is_trained:
            try:
                self.load()
            except FileNotFoundError:
                return {"status": "not_trained"}

        return {
            "status": "trained",
            "model_type": "Ridge Regression + Polynomial Features",
            **self.metadata,
        }
