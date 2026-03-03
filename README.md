# 🤖 Smart Business Operations AI Agent

A production-grade AI Agent & Machine Learning system that automates data insights, business queries, and operational tasks using ML models, LangChain agents, and a Streamlit dashboard.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT DASHBOARD                       │
│         (Natural Language Input + Visualization)            │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP / WebSocket
┌───────────────────────▼─────────────────────────────────────┐
│                   FASTAPI BACKEND                            │
│   ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│   │  /query     │  │  /predict    │  │  /anomaly         │  │
│   │  /insights  │  │  /agent      │  │  /alerts          │  │
│   └──────┬──────┘  └──────┬───────┘  └─────────┬─────────┘  │
└──────────┼────────────────┼──────────────────────┼───────────┘
           │                │                      │
┌──────────▼────────────────▼──────────────────────▼───────────┐
│                    AI AGENT LAYER (LangChain)                 │
│   ┌──────────────┐  ┌─────────────────┐  ┌────────────────┐  │
│   │  Query Tool  │  │  Predict Tool   │  │  Anomaly Tool  │  │
│   │  (DB Query)  │  │  (ML Model)     │  │  (IsolForest)  │  │
│   └──────┬───────┘  └────────┬────────┘  └───────┬────────┘  │
│          │                   │                    │           │
│   ┌──────▼───────────────────▼────────────────────▼────────┐  │
│   │           LOCAL LLM (Ollama + Mistral)                 │  │
│   │           + Conversation Memory                        │  │
│   └────────────────────────────────────────────────────────┘  │
└──────────┬──────────────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────────┐
│                    DATA LAYER                                │
│   ┌────────────────┐     ┌─────────────────────────────┐    │
│   │  SQLite / PG   │     │   FAISS Vector Store         │    │
│   │  (Sales Data)  │     │   (Business Documents RAG)  │    │
│   └────────────────┘     └─────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────────┐
│                    ML MODEL LAYER                            │
│   ┌────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│   │  Sales         │  │  Anomaly        │  │  Trend      │  │
│   │  Forecasting   │  │  Detection      │  │  Analysis   │  │
│   │  (LinearReg)   │  │  (IsolForest)   │  │  (Stats)    │  │
│   └────────────────┘  └─────────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Folder Structure

```
smart_business_ai/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app entry point
│   │   ├── routes/
│   │   │   ├── query.py         # NL query endpoints
│   │   │   ├── predict.py       # ML prediction endpoints
│   │   │   ├── anomaly.py       # Anomaly detection endpoints
│   │   │   └── insights.py      # Auto insights endpoints
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── business_agent.py    # Main LangChain agent
│   │   ├── tools.py             # Agent tools (DB, ML, etc.)
│   │   └── memory.py            # Conversation memory
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── forecasting.py       # Sales forecasting model
│   │   ├── anomaly.py           # Anomaly detection model
│   │   └── model_manager.py     # Model persistence
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py        # DB connection
│   │   ├── models.py            # SQLAlchemy models
│   │   ├── queries.py           # Business query functions
│   │   └── seed_data.py         # Sample data generator
│   └── utils/
│       ├── __init__.py
│       ├── alerts.py            # Alert system
│       ├── insights.py          # Insight generation
│       └── logger.py            # Logging setup
├── ml_models/
│   ├── trained/                 # Saved .pkl models
│   └── data/                    # Training datasets
├── dashboard/
│   ├── app.py                   # Main Streamlit app
│   ├── pages/
│   │   ├── overview.py          # KPI overview
│   │   ├── predictions.py       # Sales predictions
│   │   ├── anomalies.py         # Anomaly view
│   │   └── agent_chat.py        # Chat interface
│   └── components/
│       ├── charts.py            # Reusable chart components
│       └── metrics.py           # Metric cards
├── scripts/
│   ├── train_models.py          # Model training script
│   ├── seed_database.py         # DB seeding script
│   └── setup_rag.py             # RAG vector store setup
├── tests/
│   ├── test_api.py
│   ├── test_ml.py
│   └── test_agent.py
├── config/
│   ├── settings.py              # App configuration
│   └── prompts.py               # LLM prompt templates
├── docs/
│   └── business_docs/           # Sample business docs for RAG
├── .env.example
├── requirements.txt
├── docker-compose.yml
├── Procfile                     # For Render deployment
└── README.md
```

---

## 🛠️ Tech Stack & Justification

| Layer | Technology | Why |
|-------|-----------|-----|
| Backend | FastAPI | Async, fast, auto-docs, production-ready |
| ML | Scikit-learn | Battle-tested, lightweight, easy persistence |
| AI Agent | LangChain | Tool-based agents, memory, RAG support |
| LLM | Ollama + Mistral | 100% free, runs locally, no API costs |
| Database | SQLite (dev) / PostgreSQL (prod) | Zero setup for dev, scalable for prod |
| Vector DB | FAISS | Free, fast, works offline |
| Dashboard | Streamlit | Fast Python-native dashboards |
| Deployment | Render / Railway | Free tier available |

---

## 🚀 Quick Start

### 1. Prerequisites
```bash
# Install Ollama (for local LLM)
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull mistral

# Python 3.10+
python --version
```

### 2. Setup Environment
```bash
git clone <your-repo>
cd smart_business_ai

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure
```bash
cp .env.example .env
# Edit .env with your settings
```

### 4. Initialize Database & Train Models
```bash
python scripr

ts/seed_database.py
python scripts/train_models.py
python scripts/setup_rag.py  # Optional: for RAG features
```

### 5. Start Backend
```bash
cd backend/api
uvicorn main:app --reload --port 8000
```

### 6. Start Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/agent/query` | Natural language query |
| POST | `/api/predict/sales` | Sales forecasting |
| GET | `/api/anomaly/detect` | Run anomaly detection |
| GET | `/api/insights/summary` | Auto-generate insights |
| GET | `/api/data/metrics` | Get KPI metrics |
| POST | `/api/alerts/check` | Check for alerts |

---

## 🎤 Interview Explanation

**"Tell me about this project"**

> "I built a production-grade AI automation system called Smart Business Operations AI Agent. The system accepts natural language business queries — like 'predict next quarter revenue' or 'detect anomalies in March sales' — and routes them through a LangChain agent that decides which tools to call. The agent can query a structured database, call trained scikit-learn models for forecasting and anomaly detection, or retrieve context from business documents using RAG with FAISS. The LLM backbone is Mistral running locally via Ollama — completely free, no API costs. Everything is exposed through a FastAPI backend with clean REST endpoints, and visualized on a Streamlit dashboard. I designed it with model persistence, conversation memory, automated alerting, and a clean modular architecture that can scale to PostgreSQL and cloud deployment."

**Key talking points:**
- Multi-tool LangChain agent with routing logic
- Production patterns: model persistence, structured logging, env config
- Local LLM for cost-zero inference
- RAG pipeline for document Q&A
- End-to-end from data → model → API → UI

---

## 🔮 Future Enhancements

1. **Replace Mistral with GPT-4** via OpenAI API for higher accuracy
2. **Add Apache Kafka** for real-time streaming anomaly detection
3. **MLflow** for experiment tracking and model versioning
4. **Celery + Redis** for async background task processing
5. **Authentication** via JWT tokens
6. **Multi-tenant** support for different business units
7. **Time-series models** (Prophet, ARIMA) for better forecasting
8. **Slack/Email alerts** via webhook integrations
9. **Kubernetes deployment** with auto-scaling
10. **A/B testing** framework for model comparison
