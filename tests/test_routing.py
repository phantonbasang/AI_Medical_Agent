"""
tests/test_routing.py

Unit tests for the supervisor routing logic in src/agents/supervisor.py

Tests cover:
- has_emergency_keywords()   : Keyword detection without LLM
- supervisor_node()          : State mutation and routing decisions
- Route fallback behaviour   : LLM failure should default to rag_agent
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage


# ---------------------------------------------------------------------------
# has_emergency_keywords
# ---------------------------------------------------------------------------

class TestEmergencyKeywords:
    """Tests for the keyword-based emergency detection."""

    @pytest.mark.parametrize("text", [
        "I have chest pain",
        "He cannot breathe at all",
        "This is a stroke",
        "Patient is unconscious",
        "Suspected overdose",
    ])
    def test_detects_emergency_keywords(self, text: str):
        from src.agents.supervisor import has_emergency_keywords
        assert has_emergency_keywords(text) is True

    @pytest.mark.parametrize("text", [
        "I have a mild headache",
        "What is diabetes?",
        "Tell me about metformin",
        "Diet tips for hypertension",
        "I feel a bit tired today",
    ])
    def test_no_false_positives(self, text: str):
        from src.agents.supervisor import has_emergency_keywords
        assert has_emergency_keywords(text) is False

    def test_case_insensitive(self):
        from src.agents.supervisor import has_emergency_keywords
        assert has_emergency_keywords("CHEST PAIN RIGHT NOW") is True
        assert has_emergency_keywords("Chest Pain") is True


# ---------------------------------------------------------------------------
# supervisor_node routing
# ---------------------------------------------------------------------------

def _build_state(message: str) -> dict:
    return {"messages": [HumanMessage(content=message)]}


class TestSupervisorNode:
    """Tests for the supervisor_node LangGraph node."""

    def test_emergency_keyword_bypasses_llm(self):
        """Emergency keyword detection must route without calling the LLM."""
        from src.agents.supervisor import supervisor_node

        # No LLM mock needed — keyword bypass fires before LLM call
        result = supervisor_node(_build_state("I have severe chest pain!"))

        assert result["route_to"] == "emergency_agent"
        assert result["is_emergency"] is True
        assert result["urgency_level"] == 5

    def test_llm_routes_to_rag(self):
        """General question should route to rag_agent via LLM classification."""
        llm_response = json.dumps({
            "intent": "rag_query",
            "route_to": "rag_agent",
            "urgency_level": 1,
            "is_emergency": False,
            "reasoning": "General info query.",
        })
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=llm_response)

        with patch("src.agents.supervisor.get_llm", return_value=mock_llm):
            from src.agents.supervisor import supervisor_node
            result = supervisor_node(_build_state("What is Type 2 Diabetes?"))

        assert result["route_to"] == "rag_agent"
        assert result["is_emergency"] is False

    def test_llm_routes_to_symptom_agent(self):
        """Symptom query should be routed to symptom_agent."""
        llm_response = json.dumps({
            "intent": "symptom_query",
            "route_to": "symptom_agent",
            "urgency_level": 3,
            "is_emergency": False,
            "reasoning": "User describes symptoms.",
        })
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=llm_response)

        with patch("src.agents.supervisor.get_llm", return_value=mock_llm):
            from src.agents.supervisor import supervisor_node
            result = supervisor_node(_build_state("I've had a headache for 3 days."))

        assert result["route_to"] == "symptom_agent"

    def test_llm_routes_to_drug_agent(self):
        """Drug query should be routed to drug_agent."""
        llm_response = json.dumps({
            "intent": "drug_query",
            "route_to": "drug_agent",
            "urgency_level": 1,
            "is_emergency": False,
            "reasoning": "User asks about medication.",
        })
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=llm_response)

        with patch("src.agents.supervisor.get_llm", return_value=mock_llm):
            from src.agents.supervisor import supervisor_node
            result = supervisor_node(_build_state("What are the side effects of metformin?"))

        assert result["route_to"] == "drug_agent"

    def test_llm_routes_to_lifestyle_agent(self):
        """Lifestyle / prevention query should be routed to lifestyle_agent."""
        llm_response = json.dumps({
            "intent": "lifestyle_query",
            "route_to": "lifestyle_agent",
            "urgency_level": 1,
            "is_emergency": False,
            "reasoning": "User asks about diet.",
        })
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=llm_response)

        with patch("src.agents.supervisor.get_llm", return_value=mock_llm):
            from src.agents.supervisor import supervisor_node
            result = supervisor_node(_build_state("What foods should I avoid with high blood pressure?"))

        assert result["route_to"] == "lifestyle_agent"

    def test_high_urgency_overrides_to_emergency(self):
        """Urgency level >= threshold should force emergency_agent even without keywords."""
        llm_response = json.dumps({
            "intent": "symptom_query",
            "route_to": "symptom_agent",
            "urgency_level": 5,
            "is_emergency": False,  # LLM did not mark emergency
            "reasoning": "Very severe symptoms.",
        })
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=llm_response)

        with patch("src.agents.supervisor.get_llm", return_value=mock_llm):
            from src.agents.supervisor import supervisor_node
            result = supervisor_node(_build_state("I am experiencing extreme symptoms."))

        # Urgency 5 >= threshold (4) → emergency override
        assert result["is_emergency"] is True
        assert result["route_to"] == "emergency_agent"

    def test_llm_failure_defaults_to_rag(self):
        """When the LLM call fails, fall back to rag_agent gracefully."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API timeout")

        with patch("src.agents.supervisor.get_llm", return_value=mock_llm):
            from src.agents.supervisor import supervisor_node
            result = supervisor_node(_build_state("Tell me about hypertension."))

        assert result["route_to"] == "rag_agent"
        assert result["is_emergency"] is False

    def test_empty_messages_returns_safe_default(self):
        """Supervisor with no messages should not crash."""
        from src.agents.supervisor import supervisor_node
        result = supervisor_node({"messages": []})
        assert "route_to" in result
        assert result["route_to"] == "rag_agent"

    def test_preserves_existing_state_fields(self):
        """Supervisor should not discard unrelated state fields."""
        llm_response = json.dumps({
            "intent": "rag_query",
            "route_to": "rag_agent",
            "urgency_level": 1,
            "is_emergency": False,
            "reasoning": "General query.",
        })
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=llm_response)

        with patch("src.agents.supervisor.get_llm", return_value=mock_llm):
            from src.agents.supervisor import supervisor_node
            state = _build_state("What is diabetes?")
            state["session_id"] = "test-session-123"
            result = supervisor_node(state)

        assert result.get("session_id") == "test-session-123"
