"""
src/agents/graph.py

LangGraph assembly for the AI Medical Agent.

Architecture
------------

                ┌───────────────────────┐
                │   supervisor_node     │  ← Intent router
                └─────────┬─────────────┘
                          │ route_to
          ┌───────────────┼───────────────┬──────────────┐
          ▼               ▼               ▼              ▼
     rag_agent    symptom_agent     drug_agent   lifestyle_agent
          │               │               │              │
          └───────────────┴───────────────┴──────────────┘
                          │
                   emergency_agent  (instant bypass if is_emergency)
                          │
                         END

Usage
-----
    from src.agents.graph import build_medical_graph
    graph = build_medical_graph()

    result = graph.invoke(
        {"messages": [HumanMessage("What is diabetes?")]},
        config={"configurable": {"thread_id": "session-1"}},
    )
    print(result["agent_response"])
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from src.agents.supervisor import MedicalAgentState, RouteType, supervisor_node
from src.agents.tools import (
    DRUG_TOOLS,
    LIFESTYLE_TOOLS,
    RAG_TOOLS,
    SYMPTOM_TOOLS,
)
from src.models.llm import get_llm
from src.utils import get_logger, safe_extract_content

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# System prompts for each specialist agent
# ---------------------------------------------------------------------------

_RAG_SYSTEM = """You are a medical research assistant specialising in evidence-based medicine.
Use the search_knowledge_base and explain_medical_term tools to retrieve accurate information.
Always cite the source of your information and remind users to consult a healthcare professional."""

_SYMPTOM_SYSTEM = """You are a medical triage assistant.
Use the assess_urgency tool to evaluate the severity of reported symptoms.
Provide clear, calm guidance and always recommend professional medical consultation.
Never make a definitive diagnosis."""

_DRUG_SYSTEM = """You are a clinical pharmacist assistant.
Use the get_drug_info and search_knowledge_base tools to provide accurate medication information.
Always mention that dosage and treatment decisions must be made by a licensed healthcare provider."""

_LIFESTYLE_SYSTEM = """You are a preventive health and wellness advisor.
Use the get_lifestyle_advice and search_knowledge_base tools to provide evidence-based recommendations.
Focus on sustainable lifestyle changes and always encourage regular check-ups."""

_EMERGENCY_RESPONSE = """🚨 **MEDICAL EMERGENCY DETECTED**

Please take the following steps IMMEDIATELY:

1. **Call emergency services** (911 in the US, 115 in Vietnam, 999 in the UK)
2. **Stay calm** and keep the patient still
3. **Do not give food or water** unless instructed by emergency services
4. **Stay on the line** with emergency services until help arrives

⚠️ This AI system cannot provide emergency medical assistance.
**Call for professional emergency help right now.**"""


# ---------------------------------------------------------------------------
# Agent node factory
# ---------------------------------------------------------------------------

def _make_agent_node(system_prompt: str, tools: list, node_name: str):
    """
    Create a LangGraph node function that wraps a ReAct agent.

    The node:
    1. Runs the ReAct agent with the provided tools.
    2. Extracts the final AI response text.
    3. Appends it to ``state["messages"]`` and sets ``state["agent_response"]``.

    Parameters
    ----------
    system_prompt:
        Instructions prepended to every agent call.
    tools:
        LangChain tool objects available to this agent.
    node_name:
        Human-readable name used in log messages.

    Returns
    -------
    Callable[[MedicalAgentState], MedicalAgentState]
    """
    llm = get_llm()
    react_agent = create_react_agent(llm, tools=tools, state_modifier=system_prompt)

    def node(state: MedicalAgentState) -> MedicalAgentState:
        logger.info("[%s] Invoked.", node_name)
        try:
            result = react_agent.invoke({"messages": state.get("messages", [])})
            # Extract text from the last AI message
            agent_messages = result.get("messages", [])
            response_text = ""
            sources: list[str] = []
            for msg in reversed(agent_messages):
                if isinstance(msg, AIMessage):
                    response_text = safe_extract_content(msg)
                    break

            # Collect source metadata from tool messages
            for msg in agent_messages:
                src = getattr(msg, "name", None)
                if src and src not in sources:
                    sources.append(src)

            logger.info("[%s] Response: %s…", node_name, response_text[:80])
            return {
                **state,
                "messages": [AIMessage(content=response_text)],
                "agent_response": response_text,
                "sources": sources,
            }
        except Exception as exc:
            logger.error("[%s] Agent failed: %s", node_name, exc)
            fallback = (
                "I encountered an issue retrieving information. "
                "Please consult a qualified healthcare professional."
            )
            return {
                **state,
                "messages": [AIMessage(content=fallback)],
                "agent_response": fallback,
                "sources": [],
            }

    node.__name__ = node_name
    return node


# ---------------------------------------------------------------------------
# Emergency node (no LLM — instant)
# ---------------------------------------------------------------------------

def emergency_node(state: MedicalAgentState) -> MedicalAgentState:
    """
    Immediately return the emergency response without calling any LLM.

    This node is designed to respond in < 10 ms.
    """
    logger.warning("🚨 Emergency node activated.")
    return {
        **state,
        "messages": [AIMessage(content=_EMERGENCY_RESPONSE)],
        "agent_response": _EMERGENCY_RESPONSE,
        "sources": [],
        "urgency_level": 5,
        "is_emergency": True,
    }


# ---------------------------------------------------------------------------
# Router function
# ---------------------------------------------------------------------------

def _route(state: MedicalAgentState) -> RouteType:
    """Conditional edge: read route_to from state set by supervisor."""
    if state.get("is_emergency"):
        return "emergency_agent"
    route = state.get("route_to", "rag_agent")
    valid_routes = {"rag_agent", "symptom_agent", "drug_agent", "lifestyle_agent", "emergency_agent"}
    if route not in valid_routes:
        logger.warning("Unknown route '%s' — defaulting to rag_agent", route)
        return "rag_agent"  # type: ignore[return-value]
    return route  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_medical_graph(memory: MemorySaver | None = None) -> "CompiledGraph":  # type: ignore[name-defined]
    """
    Assemble and compile the full medical agent LangGraph.

    Parameters
    ----------
    memory:
        LangGraph checkpointer for session persistence.  If ``None``,
        an in-process ``MemorySaver`` is used (state lost on restart).

    Returns
    -------
    CompiledGraph
        Ready-to-invoke LangGraph.  Call with::

            graph.invoke(
                {"messages": [HumanMessage("...")]},
                config={"configurable": {"thread_id": "session-xyz"}},
            )
    """
    if memory is None:
        memory = MemorySaver()

    # Build specialist agent nodes
    rag_node = _make_agent_node(_RAG_SYSTEM, RAG_TOOLS, "rag_agent")
    symptom_node = _make_agent_node(_SYMPTOM_SYSTEM, SYMPTOM_TOOLS, "symptom_agent")
    drug_node = _make_agent_node(_DRUG_SYSTEM, DRUG_TOOLS, "drug_agent")
    lifestyle_node = _make_agent_node(_LIFESTYLE_SYSTEM, LIFESTYLE_TOOLS, "lifestyle_agent")

    # Assemble graph
    builder = StateGraph(MedicalAgentState)

    # Add nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("rag_agent", rag_node)
    builder.add_node("symptom_agent", symptom_node)
    builder.add_node("drug_agent", drug_node)
    builder.add_node("lifestyle_agent", lifestyle_node)
    builder.add_node("emergency_agent", emergency_node)

    # Entry point
    builder.set_entry_point("supervisor")

    # Conditional routing from supervisor
    builder.add_conditional_edges(
        "supervisor",
        _route,
        {
            "rag_agent": "rag_agent",
            "symptom_agent": "symptom_agent",
            "drug_agent": "drug_agent",
            "lifestyle_agent": "lifestyle_agent",
            "emergency_agent": "emergency_agent",
        },
    )

    # All specialist agents → END
    for node_name in ["rag_agent", "symptom_agent", "drug_agent", "lifestyle_agent", "emergency_agent"]:
        builder.add_edge(node_name, END)

    compiled = builder.compile(checkpointer=memory)
    logger.info("Medical graph compiled with %d nodes.", 6)
    return compiled
