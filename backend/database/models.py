"""
SQLAlchemy ORM models — the single source of truth for database schema.
"""
from sqlalchemy import (
    Column, Integer, Float, String, DateTime, Text, Boolean, Index
)
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime


class Base(DeclarativeBase):
    pass


class SalesRecord(Base):
    """Daily sales transactions."""
    __tablename__ = "sales_records"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, nullable=False, index=True)
    product_category = Column(String(100), nullable=False, index=True)
    region = Column(String(100), nullable=False, index=True)
    revenue = Column(Float, nullable=False)
    units_sold = Column(Integer, nullable=False)
    profit_margin = Column(Float, nullable=False)  # 0.0 to 1.0
    customer_count = Column(Integer, nullable=False)
    channel = Column(String(50), nullable=False)  # online, retail, wholesale
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_sales_date_region", "date", "region"),
        Index("ix_sales_date_category", "date", "product_category"),
    )


class OperationalMetric(Base):
    """Daily operational KPIs."""
    __tablename__ = "operational_metrics"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, nullable=False, index=True)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    unit = Column(String(50))  # dollars, count, percent, hours
    department = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)


class AnomalyLog(Base):
    """Log of detected anomalies for audit trail."""
    __tablename__ = "anomaly_logs"

    id = Column(Integer, primary_key=True, index=True)
    detected_at = Column(DateTime, default=datetime.utcnow, index=True)
    record_date = Column(DateTime, nullable=False)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    anomaly_score = Column(Float, nullable=False)  # -1 = anomaly (IsolationForest)
    severity = Column(String(20), nullable=False)  # low, medium, high, critical
    description = Column(Text)
    is_resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_anomaly_date_severity", "detected_at", "severity"),
    )


class AlertLog(Base):
    """Triggered alerts for monitoring."""
    __tablename__ = "alert_logs"

    id = Column(Integer, primary_key=True, index=True)
    triggered_at = Column(DateTime, default=datetime.utcnow, index=True)
    alert_type = Column(String(100), nullable=False)  # revenue_drop, anomaly, etc.
    severity = Column(String(20), nullable=False)
    message = Column(Text, nullable=False)
    metric_value = Column(Float)
    threshold_value = Column(Float)
    is_acknowledged = Column(Boolean, default=False)


class QueryLog(Base):
    """Log of user queries for analytics."""
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    user_query = Column(Text, nullable=False)
    agent_response = Column(Text)
    tools_used = Column(String(500))
    response_time_ms = Column(Integer)
    was_successful = Column(Boolean, default=True)
