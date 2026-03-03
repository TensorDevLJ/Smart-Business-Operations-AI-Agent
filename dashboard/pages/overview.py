"""Overview Dashboard Page — KPI Summary and Revenue Trends."""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render(backend_url: str):
    st.markdown('<div class="main-header">🏠 Business Operations Overview</div>', unsafe_allow_html=True)

    # ── KPI Metrics Row ──────────────────────────────────
    try:
        resp = requests.get(f"{backend_url}/api/data/metrics", timeout=5)
        if resp.status_code == 200:
            metrics = resp.json()
            render_kpi_cards(metrics)
        else:
            st.error("Failed to load KPI metrics")
    except Exception as e:
        st.error(f"Backend unavailable: {e}")
        st.info("Make sure the backend is running: `uvicorn backend.api.main:app --reload --port 8000`")
        return

    st.markdown("---")

    # ── Charts Row ───────────────────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        render_revenue_trend(backend_url)

    with col2:
        render_regional_breakdown(backend_url)

    st.markdown("---")

    # ── Category Performance ─────────────────────────────
    render_category_performance(backend_url)


def render_kpi_cards(metrics: dict):
    """Render KPI metric cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📅 Current Month Revenue",
            f"${metrics.get('current_month_revenue', 0):,.0f}",
            f"{metrics.get('mom_change_pct', 0):+.1f}% vs last month",
        )

    with col2:
        st.metric(
            "📊 Last Month Revenue",
            f"${metrics.get('last_month_revenue', 0):,.0f}",
        )

    with col3:
        st.metric(
            "📈 Year-to-Date Revenue",
            f"${metrics.get('ytd_revenue', 0):,.0f}",
        )

    with col4:
        anomaly_count = metrics.get('active_anomalies', 0)
        delta_color = "inverse" if anomaly_count > 0 else "normal"
        st.metric(
            "⚠️ Active Anomalies",
            anomaly_count,
            delta_color=delta_color,
        )


def render_revenue_trend(backend_url: str):
    """Revenue trend line chart."""
    st.markdown("#### 📈 Monthly Revenue Trend (12 months)")

    try:
        resp = requests.get(f"{backend_url}/api/data/monthly-revenue?months=12", timeout=5)
        if resp.status_code != 200:
            st.warning("Could not load revenue data")
            return

        data = resp.json()["data"]
        if not data:
            st.info("No revenue data available")
            return

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["period"])

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=df["period"],
                y=df["total_revenue"],
                name="Revenue",
                line=dict(color="#3b82f6", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(59, 130, 246, 0.1)",
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=df["period"],
                y=df["avg_margin"],
                name="Margin %",
                line=dict(color="#22c55e", width=2, dash="dot"),
            ),
            secondary_y=True,
        )

        fig.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            plot_bgcolor="white",
            paper_bgcolor="white",
            hovermode="x unified",
        )
        fig.update_yaxes(title_text="Revenue ($)", secondary_y=False, tickformat="$,.0f")
        fig.update_yaxes(title_text="Margin (%)", secondary_y=True, ticksuffix="%")

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Chart error: {e}")


def render_regional_breakdown(backend_url: str):
    """Regional revenue pie chart."""
    st.markdown("#### 🌍 Revenue by Region")

    try:
        resp = requests.get(f"{backend_url}/api/data/regional?months=3", timeout=5)
        if resp.status_code != 200:
            return

        data = resp.json()["data"]
        if not data:
            st.info("No regional data")
            return

        df = pd.DataFrame(data)

        fig = px.pie(
            df,
            names="region",
            values="total_revenue",
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.4,
        )
        fig.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="v", x=1.0),
            showlegend=True,
        )
        fig.update_traces(textinfo="percent+label", textfont_size=11)

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Chart error: {e}")


def render_category_performance(backend_url: str):
    """Category performance bar chart."""
    st.markdown("#### 📦 Revenue by Product Category")

    try:
        resp = requests.get(f"{backend_url}/api/data/categories?months=3", timeout=5)
        if resp.status_code != 200:
            return

        data = resp.json()["data"]
        if not data:
            return

        df = pd.DataFrame(data)
        df = df.sort_values("total_revenue", ascending=True)

        fig = px.bar(
            df,
            x="total_revenue",
            y="category",
            orientation="h",
            color="total_revenue",
            color_continuous_scale="Blues",
            text_auto=".2s",
        )
        fig.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_showscale=False,
            plot_bgcolor="white",
            xaxis_tickformat="$,.0f",
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Chart error: {e}")
