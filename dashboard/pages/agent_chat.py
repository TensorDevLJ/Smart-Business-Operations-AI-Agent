"""AI Agent Chat Interface Dashboard Page."""
import streamlit as st
import requests
import time


EXAMPLE_QUERIES = [
    "Predict next 3 months revenue",
    "Show last quarter performance",
    "Detect any anomalies in operations",
    "What is our current month-over-month growth?",
    "Which region is performing best?",
    "Why did sales drop in March?",
    "Give me a full business summary",
    "What are our top performing product categories?",
]


def render(backend_url: str):
    st.markdown('<div class="main-header">🤖 AI Agent Chat</div>', unsafe_allow_html=True)

    # Check agent status
    try:
        status_resp = requests.get(f"{backend_url}/api/agent/status", timeout=3)
        if status_resp.status_code == 200:
            status = status_resp.json()
            if status.get("llm_available"):
                st.success(f"✅ AI Agent Online — Using Mistral LLM with {status['tools_count']} tools")
            else:
                st.warning(
                    "⚡ AI Agent running in **fallback mode** (LLM offline). "
                    "Start Ollama for full AI reasoning: `ollama serve && ollama pull mistral`"
                )
    except Exception:
        st.error("Backend offline. Start: `uvicorn backend.api.main:app --reload`")
        return

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": (
                "👋 Hello! I'm your Smart Business AI Agent. I can help you with:\n\n"
                "- **Revenue predictions** (ML-powered forecasting)\n"
                "- **Anomaly detection** (Isolation Forest model)\n"
                "- **Business insights** (automated analysis)\n"
                "- **Data queries** (sales, regions, categories)\n\n"
                "Try asking: *'Predict next 3 months revenue'* or *'Detect anomalies'*"
            ),
            "tools_used": [],
            "response_time_ms": 0,
        })

    # Example queries
    st.markdown("**Quick queries:**")
    cols = st.columns(4)
    for i, query in enumerate(EXAMPLE_QUERIES[:4]):
        with cols[i]:
            if st.button(query, key=f"quick_{i}", use_container_width=True):
                st.session_state.pending_query = query

    cols2 = st.columns(4)
    for i, query in enumerate(EXAMPLE_QUERIES[4:8]):
        with cols2[i]:
            if st.button(query, key=f"quick_{i+4}", use_container_width=True):
                st.session_state.pending_query = query

    st.markdown("---")

    # Chat messages
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(msg["content"])
                    if msg.get("tools_used"):
                        tools_str = " → ".join(msg["tools_used"])
                        ms = msg.get("response_time_ms", 0)
                        llm_badge = "🧠 LLM" if msg.get("llm_used") else "⚡ Fallback"
                        st.caption(f"Tools: `{tools_str}` • {ms}ms • {llm_badge}")

    # Input
    user_input = st.chat_input("Ask a business question...")

    # Handle pending query from button clicks
    if "pending_query" in st.session_state:
        user_input = st.session_state.pending_query
        del st.session_state.pending_query

    if user_input:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
        })

        # Get agent response
        with st.spinner("🤖 Agent thinking..."):
            try:
                resp = requests.post(
                    f"{backend_url}/api/agent/query",
                    json={"query": user_input},
                    timeout=120,
                )

                if resp.status_code == 200:
                    result = resp.json()
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result["response"],
                        "tools_used": result.get("tools_used", []),
                        "response_time_ms": result.get("response_time_ms", 0),
                        "llm_used": result.get("llm_used", False),
                    })
                else:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"⚠️ Error: {resp.text}",
                        "tools_used": [],
                    })

            except requests.exceptions.Timeout:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "⏱️ Request timed out. The LLM may be slow to start. Try again.",
                    "tools_used": [],
                })
            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"❌ Connection error: {e}",
                    "tools_used": [],
                })

        st.rerun()

    # Controls
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            try:
                requests.delete(f"{backend_url}/api/agent/memory", timeout=3)
            except Exception:
                pass
            st.rerun()
