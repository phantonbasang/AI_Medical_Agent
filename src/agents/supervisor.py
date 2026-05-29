"""
src/agents/supervisor.py

Supervisor (router) for the AI Medical Agent LangGraph.

Responsibilities
----------------
1. Parse the user's latest message to determine intent.
2. Detect medical emergencies instantly via keyword matching (no LLM call).
3. Call the LLM to classify intent into one of the known routes.
4. Populate the shared ``MedicalAgentState`` with routing metadata.

State schema (MedicalAgentState) is defined here and imported by graph.py.

Usage
-----
    from src.agents.supervisor import supervisor_node, MedicalAgentState
"""
from __future__ import annotations

import json
import re
from typing import Annotated, Literal

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from src.utils import get_logger, load_config, safe_extract_content

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Route literals
# ---------------------------------------------------------------------------
RouteType = Literal[
    "rag_agent",
    "symptom_agent",
    "drug_agent",
    "lifestyle_agent",
    "emergency_agent",
    "end",
]

# ---------------------------------------------------------------------------
# Shared state schema
# ---------------------------------------------------------------------------

class MedicalAgentState(TypedDict, total=False):
    """
    Shared state passed between all nodes in the LangGraph.

    Fields
    ------
    messages:
        Full conversation history (auto-merged by ``add_messages``).
    intent:
        Intent category determined by the supervisor (e.g. ``symptom_query``).
    route_to:
        Which agent node to call next.
    urgency_level:
        Triage urgency 1-5 (populated by supervisor or symptom_agent).
    is_emergency:
        ``True`` when urgency ≥ threshold or emergency keywords detected.
    agent_response:
        Final text response from the last agent node.
    sources:
        List of source strings cited in the agent response.
    session_id:
        Unique identifier for the chat session.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str
    route_to: RouteType
    urgency_level: int
    is_emergency: bool
    agent_response: str
    sources: list[str]
    session_id: str


# ---------------------------------------------------------------------------
# Emergency keyword detection (no LLM — instant)
# ---------------------------------------------------------------------------

def _load_emergency_keywords() -> list[str]:
    cfg = load_config()
    return cfg.get("agents", {}).get(
        "emergency_keywords",
        [
            "chest pain", "heart attack", "stroke", "can't breathe",
            "cannot breathe", "difficulty breathing", "severe bleeding",
            "loss of consciousness", "unconscious", "overdose", "poisoning",
            "suicide", "seizure", "anaphylaxis",
        ],
    )


def has_emergency_keywords(text: str) -> bool:
    """
    Return ``True`` if any emergency keyword is found in *text*.

    This check is O(n) string matching — deliberately no LLM call so
    it completes in < 1 ms.

    Parameters
    ----------
    text:
        User message or concatenated message history.
    """
    lowered = text.lower()
    keywords = _load_emergency_keywords()
    return any(kw in lowered for kw in keywords)


# ---------------------------------------------------------------------------
# LLM-based intent classification
# ---------------------------------------------------------------------------

_SUPERVISOR_PROMPT = """\
You are a medical AI supervisor that classifies user queries and routes them to the correct specialist agent.

Available agents:
- rag_agent        : General medical questions, disease info, medical research
- symptom_agent    : Symptom assessment, triage, "I have X symptoms"
- drug_agent       : Drug information, medications, dosages, side effects
- lifestyle_agent  : Diet, exercise, prevention, wellness, lifestyle advice
- emergency_agent  : Life-threatening emergencies (chest pain, stroke, etc.)

User message:
{user_message}

Respond ONLY with valid JSON — no markdown, no extra text:
{{
  "intent": "<symptom_query|drug_query|lifestyle_query|rag_query|emergency>",
  "route_to": "<rag_agent|symptom_agent|drug_agent|lifestyle_agent|emergency_agent>",
  "urgency_level": <1-5>,
  "is_emergency": <true|false>,
  "reasoning": "<one sentence>"
}}

Examples:
- "What is diabetes?" → rag_agent, urgency 1
- "I have chest pain and shortness of breath" → emergency_agent, urgency 5
- "What are the side effects of metformin?" → drug_agent, urgency 1
- "How should I eat with hypertension?" → lifestyle_agent, urgency 1
- "I've had a headache for 3 days" → symptom_agent, urgency 3
"""


def _classify_intent(user_message: str) -> dict:
    """Call the supervisor LLM and extract a routing JSON."""
    from src.models.llm import get_llm  # local to avoid circular import

    llm = get_llm()
    prompt = _SUPERVISOR_PROMPT.format(user_message=user_message)
    response = llm.invoke(prompt)
    raw = safe_extract_content(response)

    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"Supervisor LLM returned no JSON: {raw[:300]}")

    return json.loads(match.group())


# ---------------------------------------------------------------------------
# Supervisor node
# ---------------------------------------------------------------------------

def supervisor_node(state: MedicalAgentState) -> MedicalAgentState:
    """
    LangGraph node: Supervisor / intent router.

    1. Extract the latest human message.
    2. Run instant emergency keyword detection.
    3. If no emergency keyword, call the LLM for intent classification.
    4. Apply urgency-threshold emergency override.
    5. Update and return the state with routing metadata.

    Parameters
    ----------
    state:
        Current ``MedicalAgentState``.

    Returns
    -------
    MedicalAgentState
        Updated state with ``intent``, ``route_to``, ``urgency_level``,
        and ``is_emergency`` populated.
    """
    messages: list[BaseMessage] = state.get("messages", [])
    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = str(msg.content)
            break

    if not user_message:
        logger.warning("Supervisor: no human message found in state.")
        return {
            **state,
            "intent": "rag_query",
            "route_to": "rag_agent",
            "urgency_level": 1,
            "is_emergency": False,
        }

    logger.info("Supervisor processing: '%s'", user_message[:100])

    # --- Step 1: Instant emergency keyword check ---
    if has_emergency_keywords(user_message):
        logger.warning("🚨 Emergency keywords detected — bypassing LLM routing.")
        return {
            **state,
            "intent": "emergency",
            "route_to": "emergency_agent",
            "urgency_level": 5,
            "is_emergency": True,
        }

    # --- Step 2: LLM intent classification ---
    cfg = load_config()
    threshold: int = cfg.get("agents", {}).get("emergency_urgency_threshold", 4)

    try:
        decision = _classify_intent(user_message)
        urgency: int = int(decision.get("urgency_level", 1))
        is_emergency: bool = bool(decision.get("is_emergency", False)) or urgency >= threshold
        route: RouteType = "emergency_agent" if is_emergency else decision.get("route_to", "rag_agent")  # type: ignore[assignment]
        intent: str = decision.get("intent", "rag_query")

        logger.info(
            "Supervisor → intent=%s  route=%s  urgency=%d  emergency=%s",
            intent,
            route,
            urgency,
            is_emergency,
        )

        return {
            **state,
            "intent": intent,
            "route_to": route,
            "urgency_level": urgency,
            "is_emergency": is_emergency,
        }

    except Exception as exc:
        logger.error("Supervisor LLM call failed: %s — falling back to rag_agent", exc)
        return {
            **state,
            "intent": "rag_query",
            "route_to": "rag_agent",
            "urgency_level": 1,
            "is_emergency": False,
        }
