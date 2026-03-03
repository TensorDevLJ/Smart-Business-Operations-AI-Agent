# 🚀 Deployment Guide

## Option 1: Render (Free Tier) — Backend API

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "feat: initial Smart Business AI Agent"
git remote add origin https://github.com/YOUR_USERNAME/smart-business-ai
git push -u origin main
```

### Step 2: Deploy Backend on Render
1. Go to https://render.com → New Web Service
2. Connect your GitHub repo
3. Settings:
   - **Name:** smart-business-ai-backend
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt && python scripts/seed_database.py && python scripts/train_models.py`
   - **Start Command:** `uvicorn backend.api.main:app --host 0.0.0.0 --port $PORT`
4. Environment Variables:
   - `DATABASE_URL` → Render provides free PostgreSQL (add as connected DB)
   - `OLLAMA_BASE_URL` → N/A (Render doesn't support Ollama — use fallback mode)
   - `DEBUG` → `false`

### Step 3: Deploy Dashboard on Streamlit Cloud
1. Go to https://share.streamlit.io
2. Connect GitHub → select your repo
3. Main file path: `dashboard/app.py`
4. Advanced settings → Secrets:
   ```toml
   BACKEND_URL = "https://your-render-backend-url.onrender.com"
   ```

---

## Option 2: Railway (Free Tier)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

Railway auto-detects the Procfile. Add environment variables in Railway dashboard.

---

## Option 3: Docker (Self-hosted / VPS)

```bash
# Build and run all services
docker-compose up -d

# Pull Mistral model (first time)
docker exec smart_business_ollama ollama pull mistral

# Seed database
docker exec smart_business_backend python scripts/seed_database.py

# Train models
docker exec smart_business_backend python scripts/train_models.py
```

Access:
- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs

---

## Option 4: Local Development (Recommended First)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup Ollama (for LLM)
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
ollama pull mistral

# 4. Configure
cp .env.example .env
# Edit .env if needed

# 5. Initialize data
python scripts/seed_database.py
python scripts/train_models.py
python scripts/setup_rag.py  # Optional

# 6. Start backend (Terminal 1)
uvicorn backend.api.main:app --reload --port 8000

# 7. Start dashboard (Terminal 2)
streamlit run dashboard/app.py

# 8. Open browser
# Dashboard: http://localhost:8501
# API Docs:  http://localhost:8000/docs
```

---

## Notes on Ollama (LLM)

Render and Railway free tiers do not support Ollama due to memory requirements.
The system **gracefully degrades** to rule-based routing when LLM is offline.

For production LLM support:
- Use a VPS with 8GB+ RAM (e.g., DigitalOcean $24/mo)
- Or replace Ollama with Groq API (free tier, runs Mistral in cloud)

### Using Groq Instead of Ollama (Free API):
```python
# In config/settings.py, add:
groq_api_key: str = ""

# In business_agent.py, replace Ollama with:
from langchain_groq import ChatGroq
llm = ChatGroq(model="mixtral-8x7b-32768", api_key=settings.groq_api_key)
```
