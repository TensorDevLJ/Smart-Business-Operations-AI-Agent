"""
AI Agent API Routes.

Handles natural language business queries through the LangChain agent.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from loguru import logger

from backend.agents.business_agent import business_agent
from backend.database.connection import get_db
from backend.database.models import QueryLog

router = APIRouter()


class AgentQueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, description="Natural language business query")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")


class AgentQueryResponse(BaseModel):
    response: str
    tools_used: list
    response_time_ms: int
    llm_used: bool
    query: str


@router.post("/query", response_model=AgentQueryResponse)
async def agent_query(request: AgentQueryRequest):
    """
    Process a natural language business query using the AI agent.

    Examples:
    - "Predict next 3 months revenue"
    - "Why did sales drop in March?"
    - "Show last quarter performance"
    - "Detect any anomalies in operations"
    """
    logger.info(f"Agent query received: '{request.query[:80]}...'")

    result = business_agent.query(request.query)

    # Log to database
    try:
        with get_db() as db:
            log = QueryLog(
                user_query=request.query,
                agent_response=result["response"][:2000],
                tools_used=", ".join(result["tools_used"]),
                response_time_ms=result["response_time_ms"],
                was_successful=True,
            )
            db.add(log)
    except Exception as e:
        logger.warning(f"Failed to log query: {e}")

    return AgentQueryResponse(
        response=result["response"],
        tools_used=result["tools_used"],
        response_time_ms=result["response_time_ms"],
        llm_used=result["llm_used"],
        query=request.query,
    )


@router.delete("/memory")
async def clear_agent_memory():
    """Clear the agent's conversation memory."""
    business_agent.clear_memory()
    return {"status": "success", "message": "Agent memory cleared"}


@router.get("/status")
async def agent_status():
    """Get current agent status."""
    return {
        "llm_available": business_agent.is_llm_available,
        "model": "mistral" if business_agent.is_llm_available else "fallback (rule-based)",
        "tools_count": len(business_agent.tools),
        "tools": [t.name for t in business_agent.tools],
    }
