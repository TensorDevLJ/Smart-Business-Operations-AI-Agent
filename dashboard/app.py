"""
Smart Business Operations AI Agent — Streamlit Dashboard.

Main entry point. Handles navigation and shared state.
Run: streamlit run dashboard/app.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st

# Page config MUST be first Streamlit call
st.set_page_config(
    page_title="Smart Business AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

import requests
from datetime import datetime

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3b82f6;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .alert-critical { border-left-color: #ef4444; }
    .alert-high { border-left-color: #f97316; }
    .alert-medium { border-left-color: #eab308; }
    .alert-info { border-left-color: #3b82f6; }
    .status-dot-green { color: #22c55e; font-size: 0.8rem; }
    .status-dot-red { color: #ef4444; font-size: 0.8rem; }
    .agent-message-user {
        background: #dbeafe;
        border-radius: 12px 12px 2px 12px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        margin-left: 20%;
    }
    .agent-message-ai {
        background: #f0fdf4;
        border-radius: 12px 12px 12px 2px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        margin-right: 20%;
        border-left: 3px solid #22c55e;
    }
    .sidebar-nav-item {
        padding: 0.4rem 0;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Backend Health Check
# ─────────────────────────────────────────────

def check_backend_health():
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


# ─────────────────────────────────────────────
#  Sidebar Navigation
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🤖 Smart Business AI")
    st.markdown("---")

    # Health status
    health = check_backend_health()
    if health:
        st.markdown('<span class="status-dot-green">●</span> Backend Connected', unsafe_allow_html=True)
        llm_status = health.get("components", {}).get("llm", "unknown")
        if "available" in llm_status:
            st.markdown('<span class="status-dot-green">●</span> LLM Online (Mistral)', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-dot-red">●</span> LLM Offline (Fallback)', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-dot-red">●</span> Backend Offline', unsafe_allow_html=True)
        st.warning("Start backend: `uvicorn backend.api.main:app --reload`")

    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📈 Predictions", "🚨 Anomalies", "💡 Insights", "🤖 AI Agent Chat"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Quick Stats**")

    if health:
        try:
            metrics_resp = requests.get(f"{BACKEND_URL}/api/data/metrics", timeout=3)
            if metrics_resp.status_code == 200:
                metrics = metrics_resp.json()
                st.metric("MoM Change", f"{metrics.get('mom_change_pct', 0):+.1f}%")
                st.metric("Active Anomalies", metrics.get('active_anomalies', 0))
        except Exception:
            pass

    st.markdown("---")
    st.markdown(f"*Updated: {datetime.now().strftime('%H:%M:%S')}*")
    if st.button("🔄 Refresh"):
        st.rerun()


# ─────────────────────────────────────────────
#  Page Routing
# ─────────────────────────────────────────────

if page == "🏠 Overview":
    from dashboard.pages import overview
    overview.render(BACKEND_URL)

elif page == "📈 Predictions":
    from dashboard.pages import predictions
    predictions.render(BACKEND_URL)

elif page == "🚨 Anomalies":
    from dashboard.pages import anomalies
    anomalies.render(BACKEND_URL)

elif page == "💡 Insights":
    from dashboard.pages import insights_page
    insights_page.render(BACKEND_URL)

elif page == "🤖 AI Agent Chat":
    from dashboard.pages import agent_chat
    agent_chat.render(BACKEND_URL)
