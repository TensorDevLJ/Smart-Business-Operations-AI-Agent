"""
Business AI Agent — the core intelligence of the system.

Architecture:
- LangChain ReAct agent (Reason + Act)
- Tool-based routing: agent decides which tools to call
- Conversation memory: maintains context across turns
- Local LLM: Ollama + Mistral (no API cost)
- Fallback: works without LLM using direct tool routing

ReAct flow:
  User Query → Agent thinks → Selects tool → Gets result → Thinks → Responds
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from loguru import logger

from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

from backend.agents.tools import (
    query_database,
    predict_sales,
    detect_anomalies,
    get_kpi_metrics,
)
from config.settings import settings
from config.prompts import BUSINESS_AGENT_SYSTEM_PROMPT


# ─────────────────────────────────────────────
#  ReAct Prompt Template
# ─────────────────────────────────────────────

REACT_PROMPT = PromptTemplate.from_template("""
{system_prompt}

You have access to these tools:
{tools}

Use this EXACT format for every response:

Question: the input question you must answer
Thought: think about what to do
Action: the action to take (must be one of [{tool_names}])
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Previous conversation:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}
""")


class BusinessAIAgent:
    """
    Production AI Agent for business operations Q&A.

    Supports:
    - Natural language business queries
    - Multi-step reasoning (chain multiple tools)
    - Conversation memory (remembers context)
    - Graceful degradation (works even if LLM is offline)
    """

    def __init__(self):
        self.tools = [
            query_database,
            predict_sales,
            detect_anomalies,
            get_kpi_metrics,
        ]
        self._agent_executor: Optional[AgentExecutor] = None
        self._memory: Optional[ConversationBufferWindowMemory] = None
        self._llm_available = False
        self._initialize()

    def _initialize(self):
        """Initialize LLM and agent. Fails gracefully if Ollama is offline."""
        try:
            logger.info(f"Connecting to Ollama at {settings.ollama_base_url}...")

            llm = Ollama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=settings.llm_temperature,
                timeout=30,
            )

            # Test connection
            llm.invoke("Say 'OK' only.")
            self._llm_available = True
            logger.success(f"✅ Connected to Ollama ({settings.ollama_model})")

            self._memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                k=10,  # Keep last 10 turns
                return_messages=False,
            )

            prompt = REACT_PROMPT.partial(
                system_prompt=BUSINESS_AGENT_SYSTEM_PROMPT
            )

            agent = create_react_agent(
                llm=llm,
                tools=self.tools,
                prompt=prompt,
            )

            self._agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                memory=self._memory,
                verbose=settings.debug,
                max_iterations=6,
                handle_parsing_errors=True,
                return_intermediate_steps=True,
            )

        except Exception as e:
            logger.warning(
                f"Ollama not available ({e}). "
                "Agent will use direct tool routing (no LLM reasoning)."
            )
            self._llm_available = False

    def query(self, user_input: str) -> Dict:
        """
        Process a natural language business query.

        Returns:
            {
                "response": str,
                "tools_used": list,
                "response_time_ms": int,
                "llm_used": bool,
                "intermediate_steps": list,
            }
        """
        start_time = time.time()
        logger.info(f"Agent query: '{user_input[:100]}...'")

        if self._llm_available and self._agent_executor:
            return self._llm_query(user_input, start_time)
        else:
            return self._fallback_query(user_input, start_time)

    def _llm_query(self, user_input: str, start_time: float) -> Dict:
        """Full LLM-powered agent query."""
        try:
            result = self._agent_executor.invoke({"input": user_input})

            tools_used = []
            steps = result.get("intermediate_steps", [])
            for step in steps:
                if step and len(step) >= 1:
                    action = step[0]
                    tools_used.append(getattr(action, "tool", "unknown"))

            response_ms = int((time.time() - start_time) * 1000)

            return {
                "response": result.get("output", "Unable to generate response."),
                "tools_used": tools_used,
                "response_time_ms": response_ms,
                "llm_used": True,
                "intermediate_steps": [
                    {
                        "tool": getattr(step[0], "tool", ""),
                        "input": getattr(step[0], "tool_input", ""),
                        "output": str(step[1])[:300] if len(step) > 1 else "",
                    }
                    for step in steps
                ],
            }

        except Exception as e:
            logger.error(f"LLM agent error: {e}")
            return self._fallback_query(user_input, start_time)

    def _fallback_query(self, user_input: str, start_time: float) -> Dict:
        """
        Rule-based fallback when LLM is unavailable.
        Routes queries to tools based on keyword matching.
        """
        query_lower = user_input.lower()
        results = []
        tools_used = []

        # Routing rules
        if any(kw in query_lower for kw in ["predict", "forecast", "next", "future", "upcoming"]):
            months = 3
            for word in ["1 month", "2 months", "3 months", "6 months", "12 months"]:
                if word in query_lower:
                    months = int(word.split()[0])
            result = predict_sales.invoke({"months_ahead": months})
            results.append(result)
            tools_used.append("predict_sales")

        if any(kw in query_lower for kw in ["anomaly", "anomalies", "unusual", "drop", "spike", "weird", "issue"]):
            days = 30
            if "90 day" in query_lower or "quarter" in query_lower:
                days = 90
            result = detect_anomalies.invoke({"days_to_analyze": days})
            results.append(result)
            tools_used.append("detect_anomalies")

        if any(kw in query_lower for kw in ["revenue", "sales", "performance", "quarter", "monthly", "region", "category"]):
            if any(kw in query_lower for kw in ["q1", "q2", "q3", "q4", "quarter"]):
                current_year = datetime.utcnow().year
                for q in [1, 2, 3, 4]:
                    if f"q{q}" in query_lower:
                        result = query_database.invoke({"query_type": f"Q{q}_{current_year}"})
                        results.append(result)
                        tools_used.append("query_database")
                        break
            elif "region" in query_lower:
                result = query_database.invoke({"query_type": "regional_performance"})
                results.append(result)
                tools_used.append("query_database")
            elif "category" in query_lower or "product" in query_lower:
                result = query_database.invoke({"query_type": "category_performance"})
                results.append(result)
                tools_used.append("query_database")
            else:
                result = query_database.invoke({"query_type": "monthly_revenue"})
                results.append(result)
                tools_used.append("query_database")

        if any(kw in query_lower for kw in ["kpi", "summary", "overview", "dashboard", "metrics"]):
            result = get_kpi_metrics.invoke({"metric_type": "all"})
            results.append(result)
            tools_used.append("get_kpi_metrics")

        # Default fallback
        if not results:
            result = get_kpi_metrics.invoke({"metric_type": "all"})
            results.append(result)
            tools_used.append("get_kpi_metrics")
            results.append(
                "\n(Note: LLM is offline. For full AI reasoning, start Ollama: `ollama serve`)"
            )

        response_ms = int((time.time() - start_time) * 1000)

        return {
            "response": "\n\n".join(results),
            "tools_used": tools_used,
            "response_time_ms": response_ms,
            "llm_used": False,
            "intermediate_steps": [],
        }

    def clear_memory(self):
        """Reset conversation history."""
        if self._memory:
            self._memory.clear()
            logger.info("Agent memory cleared")

    @property
    def is_llm_available(self) -> bool:
        return self._llm_available


# Global singleton agent instance
business_agent = BusinessAIAgent()
