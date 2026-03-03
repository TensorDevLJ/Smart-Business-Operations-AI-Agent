"""
Model Training Script.

Trains all ML models using data from the database.
Must be run after seed_database.py.

Usage:
    python scripts/train_models.py

What it does:
1. Loads sales data from SQLite/PostgreSQL
2. Trains SalesForecastingModel (Ridge Regression)
3. Trains AnomalyDetectionModel (Isolation Forest)
4. Saves both models to ml_models/trained/
5. Prints training metrics
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from loguru import logger
from backend.database.connection import init_db, get_db
from backend.database.queries import get_sales_dataframe
from backend.ml.forecasting import SalesForecastingModel
from backend.ml.anomaly import AnomalyDetectionModel
from config.settings import settings


def train_all_models():
    logger.info("=" * 60)
    logger.info("  Smart Business AI — Model Training")
    logger.info("=" * 60)

    # Ensure DB is initialized
    init_db()

    # Load training data
    logger.info("Loading training data from database...")
    with get_db() as db:
        df = get_sales_dataframe(db, months=24)

    if df.empty:
        logger.error(
            "No training data found! Run database seeding first:\n"
            "  python scripts/seed_database.py"
        )
        sys.exit(1)

    logger.info(f"Loaded {len(df):,} records | "
                f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    # ── Train Forecasting Model ───────────────────────────
    logger.info("\n[1/2] Training Sales Forecasting Model...")
    forecasting_model = SalesForecastingModel(settings.ml_models_dir)
    forecast_metrics = forecasting_model.train(df)

    logger.success("Forecasting Model Trained:")
    logger.success(f"  Type:     {forecast_metrics.get('model_type', 'Ridge + Polynomial')}")
    logger.success(f"  Samples:  {forecast_metrics['training_samples']:,}")
    logger.success(f"  CV MAPE:  {forecast_metrics['cv_mape_mean']:.2%} ± {forecast_metrics['cv_mape_std']:.2%}")
    logger.success(f"  R² Score: {forecast_metrics['final_r2']:.4f}")
    logger.success(f"  Saved to: {settings.ml_models_dir}/sales_forecasting.pkl")

    # ── Train Anomaly Model ──────────────────────────────
    logger.info("\n[2/2] Training Anomaly Detection Model...")
    anomaly_model = AnomalyDetectionModel(settings.ml_models_dir)
    anomaly_metrics = anomaly_model.train(df)

    logger.success("Anomaly Model Trained:")
    logger.success(f"  Type:           Isolation Forest")
    logger.success(f"  Samples:        {anomaly_metrics['training_samples']:,}")
    logger.success(f"  Anomaly rate:   {anomaly_metrics['anomaly_rate']:.1%}")
    logger.success(f"  Training anomalies: {anomaly_metrics['anomalies_found_in_training']}")
    logger.success(f"  Saved to:       {settings.ml_models_dir}/anomaly_detection.pkl")

    # ── Quick Validation ─────────────────────────────────
    logger.info("\n[Validation] Running quick prediction test...")
    predictions = forecasting_model.predict(periods=90)
    logger.success(f"  ✅ Generated {len(predictions)} daily predictions (90 days)")
    logger.success(f"  First prediction: {predictions[0]['date']} → ${predictions[0]['predicted_revenue']:,.0f}")

    logger.info("\n" + "=" * 60)
    logger.success("✅ All models trained and saved successfully!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("  Start backend: uvicorn backend.api.main:app --reload --port 8000")
    logger.info("  Start dashboard: streamlit run dashboard/app.py")


if __name__ == "__main__":
    train_all_models()
