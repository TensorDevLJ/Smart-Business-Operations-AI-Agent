"""
RAG Setup Script — Build FAISS vector store from business documents.

Usage:
    python scripts/setup_rag.py

This creates a FAISS index from documents in docs/business_docs/
which the AI agent can then use for document Q&A.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path
from loguru import logger
from config.settings import settings


SAMPLE_DOCS = {
    "company_overview.txt": """
    Smart Business Operations — Company Overview

    Our company operates across 5 regions: North America, Europe, Asia Pacific,
    Latin America, and Middle East. We offer 5 product categories:
    Software, Hardware, Services, Consulting, and Support.

    Revenue model:
    - Software: 40-45% of total revenue, highest margin (70-85%)
    - Services: 25-30%, strong margin (55-70%)
    - Hardware: 15-20%, lower margin (25-40%)
    - Consulting: 10-15%, good margin (60-75%)
    - Support: 5-10%, stable margin (50-65%)

    Key business drivers:
    - Q4 is traditionally strongest (Black Friday, year-end deals)
    - Q1 shows seasonal dip, especially January
    - North America accounts for ~40% of global revenue
    - Online channel growing at 15% YoY

    Risk factors:
    - Regional concentration in North America
    - Hardware margin pressure from competition
    - Customer churn rate target: <3% monthly
    """,

    "kpi_targets_2024.txt": """
    KPI Targets 2024

    Revenue targets:
    - Annual revenue growth: 15% YoY
    - Monthly recurring revenue (MRR) growth: 5% MoM
    - Q4 revenue: 35% of annual target

    Profitability:
    - Overall gross margin target: 65%
    - Software margin: maintain >70%
    - Hardware margin: improve to >30%

    Operations:
    - Customer satisfaction score: >4.5/5.0
    - Support ticket resolution: <24 hours
    - Employee productivity: >90%
    - Customer churn: <2.5% monthly

    Anomaly thresholds (triggers immediate review):
    - Revenue drop >10% MoM
    - Margin compression >5 points
    - Customer count drop >15% week-over-week
    - Support tickets spike >50% above 30-day average
    """,

    "sales_process.txt": """
    Sales Process & Channel Strategy

    Direct Sales (30% of revenue):
    - Enterprise accounts >$100K
    - Managed by regional directors
    - Average deal cycle: 45-90 days

    Online Channel (40% of revenue):
    - SMB and mid-market focus
    - Self-service for <$10K deals
    - Growing 15% YoY

    Retail Partners (20% of revenue):
    - Hardware and packaged software
    - 250+ retail locations globally
    - Holiday season critical (Oct-Dec)

    Wholesale (10% of revenue):
    - OEM and reseller agreements
    - Stable, recurring revenue
    - Lower margin but predictable

    Regional notes:
    - North America: Direct sales dominant
    - Europe: Partner-led, compliance-sensitive
    - APAC: Online channel growing fastest
    - LATAM: Retail and wholesale focus
    - Middle East: Government contracts important
    """
}


def setup_rag():
    """Create FAISS vector store from business documents."""

    # Create sample documents if they don't exist
    docs_dir = Path(settings.business_docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in SAMPLE_DOCS.items():
        filepath = docs_dir / filename
        if not filepath.exists():
            filepath.write_text(content.strip())
            logger.info(f"Created sample doc: {filename}")

    # Try to build FAISS index
    try:
        from langchain_community.document_loaders import TextLoader, DirectoryLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings

        logger.info("Loading business documents...")
        loader = DirectoryLoader(str(docs_dir), glob="*.txt", loader_cls=TextLoader)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents")

        logger.info("Splitting documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")

        logger.info(f"Loading embedding model: {settings.embedding_model}")
        embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

        logger.info("Building FAISS index...")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        index_path = Path(settings.faiss_index_path)
        index_path.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(index_path))

        logger.success(f"✅ FAISS index saved to {index_path}")
        logger.success(f"   Indexed {len(chunks)} chunks from {len(documents)} documents")

    except ImportError as e:
        logger.warning(f"RAG dependencies not installed: {e}")
        logger.info("RAG is optional. Core functionality works without it.")
    except Exception as e:
        logger.error(f"RAG setup failed: {e}")
        logger.info("Continuing without RAG. Core features still work.")


if __name__ == "__main__":
    setup_rag()
