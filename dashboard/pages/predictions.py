"""Sales Predictions Dashboard Page."""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go


def render(backend_url: str):
    st.markdown('<div class="main-header">📈 Sales Predictions & Forecasting</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("ML-powered revenue forecasting using Ridge Regression with polynomial features and seasonal encoding.")
    with col2:
        months = st.selectbox("Forecast Horizon", [1, 3, 6, 12], index=1, format_func=lambda x: f"{x} months")

    if st.button("🔮 Generate Forecast", type="primary"):
        with st.spinner("Running ML forecasting model..."):
            try:
                resp = requests.post(
                    f"{backend_url}/api/predict/sales?months_ahead={months}",
                    timeout=30,
                )
                if resp.status_code == 200:
                    render_forecast_results(resp.json(), months, backend_url)
                elif resp.status_code == 503:
                    st.error("Model not trained. Run: `python scripts/train_models.py`")
                else:
                    st.error(f"Prediction failed: {resp.text}")
            except Exception as e:
                st.error(f"Backend error: {e}")
    else:
        render_forecast_results_placeholder(months, backend_url)


def render_forecast_results(data: dict, months: int, backend_url: str):
    """Render forecast results with confidence intervals."""
    predictions = data.get("predictions", [])
    if not predictions:
        st.warning("No predictions returned")
        return

    # Summary metrics
    total = data.get("total_predicted_revenue", 0)
    accuracy = data.get("model_accuracy", {})

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"Total {months}-Month Forecast", f"${total:,.0f}")
    with col2:
        mape = accuracy.get("mape", 0) or 0
        st.metric("Model Accuracy (MAPE)", f"{(1 - mape):.1%}")
    with col3:
        r2 = accuracy.get("r2", 0) or 0
        st.metric("R² Score", f"{r2:.4f}")

    # Forecast chart
    df = pd.DataFrame(predictions)

    # Load historical data for context
    try:
        hist_resp = requests.get(f"{backend_url}/api/data/monthly-revenue?months=6", timeout=5)
        if hist_resp.status_code == 200:
            hist_data = pd.DataFrame(hist_resp.json()["data"])
            has_historical = True
        else:
            has_historical = False
    except Exception:
        has_historical = False

    fig = go.Figure()

    # Historical data
    if has_historical:
        fig.add_trace(go.Scatter(
            x=hist_data["period"],
            y=hist_data["total_revenue"],
            name="Historical Revenue",
            line=dict(color="#3b82f6", width=2.5),
            mode="lines+markers",
            marker=dict(size=6),
        ))

    # Confidence band
    fig.add_trace(go.Scatter(
        x=df["month"].tolist() + df["month"].tolist()[::-1],
        y=df["upper_bound"].tolist() + df["lower_bound"].tolist()[::-1],
        fill="toself",
        fillcolor="rgba(34, 197, 94, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Confidence Interval",
        showlegend=True,
    ))

    # Predicted line
    fig.add_trace(go.Scatter(
        x=df["month"],
        y=df["predicted_revenue"],
        name="Predicted Revenue",
        line=dict(color="#22c55e", width=2.5, dash="dash"),
        mode="lines+markers",
        marker=dict(size=8, symbol="diamond"),
    ))

    fig.update_layout(
        title=f"Revenue Forecast — Next {months} Months",
        xaxis_title="Month",
        yaxis_title="Revenue ($)",
        yaxis_tickformat="$,.0f",
        height=420,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Monthly breakdown table
    st.markdown("#### 📊 Monthly Forecast Breakdown")
    display_df = df.rename(columns={
        "month": "Month",
        "predicted_revenue": "Predicted Revenue",
        "lower_bound": "Lower Bound",
        "upper_bound": "Upper Bound",
    })
    display_df["Predicted Revenue"] = display_df["Predicted Revenue"].apply(lambda x: f"${x:,.0f}")
    display_df["Lower Bound"] = display_df["Lower Bound"].apply(lambda x: f"${x:,.0f}")
    display_df["Upper Bound"] = display_df["Upper Bound"].apply(lambda x: f"${x:,.0f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_forecast_results_placeholder(months: int, backend_url: str):
    """Show model info before forecast is run."""
    st.info("👆 Click 'Generate Forecast' to run the ML model")

    try:
        resp = requests.get(f"{backend_url}/api/predict/model-info", timeout=3)
        if resp.status_code == 200:
            info = resp.json()
            if info.get("status") == "trained":
                st.success(f"✅ Forecasting model is trained and ready")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Model:** {info.get('model_type', 'N/A')}")
                    st.markdown(f"**Training samples:** {info.get('training_samples', 'N/A'):,}")
                with col2:
                    mape = info.get('cv_mape_mean', 0) or 0
                    st.markdown(f"**Cross-validated accuracy:** {(1-mape):.1%}")
                    st.markdown(f"**R² Score:** {info.get('final_r2', 'N/A')}")
            else:
                st.warning("Model not trained yet. Run: `python scripts/train_models.py`")
    except Exception:
        st.warning("Could not fetch model info")
