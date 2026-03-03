"""Anomaly Detection Dashboard Page."""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


SEVERITY_COLORS = {
    "critical": "#ef4444",
    "high": "#f97316",
    "medium": "#eab308",
    "low": "#3b82f6",
}

SEVERITY_ICONS = {
    "critical": "🔴",
    "high": "🟠",
    "medium": "🟡",
    "low": "🔵",
}


def render(backend_url: str):
    st.markdown('<div class="main-header">🚨 Anomaly Detection</div>', unsafe_allow_html=True)
    st.markdown("Isolation Forest ML model detects multivariate anomalies in business operations.")

    col1, col2 = st.columns([2, 1])
    with col1:
        days = st.slider("Analysis window (days)", min_value=7, max_value=365, value=90, step=7)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_detection = st.button("🔍 Run Anomaly Detection", type="primary")

    if run_detection:
        with st.spinner("Running Isolation Forest model..."):
            try:
                resp = requests.get(
                    f"{backend_url}/api/anomaly/detect?days={days}",
                    timeout=60,
                )
                if resp.status_code == 200:
                    render_anomaly_results(resp.json())
                elif resp.status_code == 503:
                    st.error("Model not trained. Run: `python scripts/train_models.py`")
                else:
                    st.error(f"Detection failed: {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("#### 📋 Anomaly History (Last 30 Days)")
    render_anomaly_history(backend_url)


def render_anomaly_results(data: dict):
    """Render fresh anomaly detection results."""
    total = data.get("total_anomalies", 0)
    breakdown = data.get("severity_breakdown", {})
    anomalies = data.get("anomalies", [])

    if total == 0:
        st.success("✅ No anomalies detected! Operations appear normal.")
        return

    # Summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Anomalies", total)
    with col2:
        st.metric("🔴 Critical", breakdown.get("critical", 0))
    with col3:
        st.metric("🟠 High", breakdown.get("high", 0))
    with col4:
        st.metric("🟡 Medium + Low", breakdown.get("medium", 0) + breakdown.get("low", 0))

    # Severity distribution chart
    if breakdown:
        fig = px.bar(
            x=list(breakdown.keys()),
            y=list(breakdown.values()),
            color=list(breakdown.keys()),
            color_discrete_map=SEVERITY_COLORS,
            title="Anomalies by Severity",
        )
        fig.update_layout(height=250, showlegend=False, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    # Anomaly list
    st.markdown("#### Detected Anomalies")
    for a in anomalies:
        severity = a.get("severity", "low")
        icon = SEVERITY_ICONS.get(severity, "⚪")
        color = SEVERITY_COLORS.get(severity, "#6b7280")

        with st.expander(
            f"{icon} [{severity.upper()}] {a['date']} — {a['description'][:80]}..."
            if len(a.get('description', '')) > 80
            else f"{icon} [{severity.upper()}] {a['date']} — {a.get('description', 'N/A')}"
        ):
            col1, col2 = st.columns(2)
            with col1:
                revenue = a.get('revenue')
                if revenue is not None:
                    st.markdown(f"**Revenue:** ${revenue:,.0f}")
                st.markdown(f"**Severity:** {severity.title()}")
            with col2:
                st.markdown(f"**Anomaly Score:** {a.get('anomaly_score', 'N/A'):.4f}")
                st.markdown(f"**Date:** {a['date']}")
            st.markdown(f"**Description:** {a.get('description', 'N/A')}")


def render_anomaly_history(backend_url: str):
    """Render historical anomalies from DB."""
    try:
        resp = requests.get(f"{backend_url}/api/anomaly/history?days=30", timeout=5)
        if resp.status_code != 200:
            st.info("No anomaly history available")
            return

        data = resp.json()
        anomalies = data.get("anomalies", [])

        if not anomalies:
            st.info("No anomalies recorded in the last 30 days")
            return

        df = pd.DataFrame(anomalies)
        df["detected_at"] = pd.to_datetime(df["detected_at"]).dt.strftime("%Y-%m-%d %H:%M")

        st.dataframe(
            df[["detected_at", "severity", "metric_name", "metric_value", "description", "is_resolved"]].rename(
                columns={
                    "detected_at": "Detected At",
                    "severity": "Severity",
                    "metric_name": "Metric",
                    "metric_value": "Value",
                    "description": "Description",
                    "is_resolved": "Resolved",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    except Exception as e:
        st.error(f"Error loading history: {e}")
