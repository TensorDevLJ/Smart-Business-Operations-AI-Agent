"""Business Insights Dashboard Page."""
import streamlit as st
import requests


SEVERITY_CONFIG = {
    "critical": ("🔴", "#fef2f2", "#ef4444"),
    "warning":  ("🟠", "#fff7ed", "#f97316"),
    "info":     ("🔵", "#eff6ff", "#3b82f6"),
    "success":  ("🟢", "#f0fdf4", "#22c55e"),
}


def render(backend_url: str):
    st.markdown('<div class="main-header">💡 Automated Business Insights</div>', unsafe_allow_html=True)
    st.markdown("AI-generated insights from pattern analysis — updated on each refresh.")

    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔄 Refresh Insights", type="primary"):
            st.rerun()

    try:
        resp = requests.get(f"{backend_url}/api/insights/summary", timeout=10)
        if resp.status_code != 200:
            st.error("Could not load insights")
            return

        data = resp.json()
        render_insights(data)

    except Exception as e:
        st.error(f"Error loading insights: {e}")

    st.markdown("---")
    render_alerts_section(backend_url)


def render_insights(data: dict):
    """Render insight cards."""
    kpis = data.get("kpi_snapshot", {})
    insights = data.get("insights", [])
    generated_at = data.get("generated_at", "")

    st.markdown(f"*Generated at: {generated_at[:19]}  •  {data.get('total_insights', 0)} insights found*")

    # KPI snapshot
    if kpis:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("YTD Revenue", f"${kpis.get('ytd_revenue', 0):,.0f}")
        with col2:
            mom = kpis.get('mom_change_pct', 0)
            st.metric("MoM Change", f"{mom:+.1f}%", delta_color="normal" if mom >= 0 else "inverse")
        with col3:
            st.metric("Active Anomalies", kpis.get("active_anomalies", 0))

    st.markdown("---")
    st.markdown("#### 📋 Key Business Insights")

    if not insights:
        st.info("No insights generated yet. Ensure the database is seeded with data.")
        return

    for i, insight in enumerate(insights):
        severity = insight.get("severity", "info")
        icon, bg_color, border_color = SEVERITY_CONFIG.get(severity, ("ℹ️", "#f9fafb", "#6b7280"))

        st.markdown(
            f"""
            <div style="
                background: {bg_color};
                border-left: 4px solid {border_color};
                border-radius: 8px;
                padding: 1rem 1.2rem;
                margin-bottom: 1rem;
            ">
                <div style="font-weight: 700; font-size: 1rem; margin-bottom: 0.3rem;">
                    {icon} {insight.get('title', 'Insight')}
                </div>
                <div style="color: #374151; margin-bottom: 0.5rem;">
                    {insight.get('description', '')}
                </div>
                <div style="color: #6b7280; font-size: 0.85rem; font-style: italic;">
                    💡 Recommendation: {insight.get('recommendation', 'N/A')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_alerts_section(backend_url: str):
    """Render recent alerts."""
    st.markdown("#### 🚨 Recent Alerts (Last 7 Days)")

    try:
        # Trigger alert check
        requests.post(f"{backend_url}/api/insights/alerts/check", timeout=5)

        resp = requests.get(f"{backend_url}/api/insights/alerts?days=7", timeout=5)
        if resp.status_code != 200:
            st.info("No alerts available")
            return

        alerts = resp.json().get("alerts", [])

        if not alerts:
            st.success("✅ No alerts in the last 7 days")
            return

        for alert in alerts[:10]:
            severity = alert.get("severity", "low")
            icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}.get(severity, "⚪")
            ack = "✓ Acknowledged" if alert.get("is_acknowledged") else "⏳ Pending"

            with st.expander(f"{icon} {alert.get('alert_type', '').replace('_', ' ').title()} — {alert.get('triggered_at', '')[:16]}"):
                st.markdown(f"**Message:** {alert.get('message', 'N/A')}")
                col1, col2 = st.columns(2)
                with col1:
                    val = alert.get('metric_value')
                    if val is not None:
                        st.markdown(f"**Value:** {val:,.2f}")
                with col2:
                    st.markdown(f"**Status:** {ack}")

    except Exception as e:
        st.error(f"Error loading alerts: {e}")
