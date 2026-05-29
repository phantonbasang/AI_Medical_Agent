"""
tests/test_emergency.py

Unit tests focused on emergency detection and the emergency_node.

Tests cover:
- has_emergency_keywords()  : All documented emergency keywords
- emergency_node()          : Response content and state correctness
- Keyword edge cases        : Case-insensitivity, partial matches
- Integration: keyword → emergency_agent routing in supervisor_node
"""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage


# ---------------------------------------------------------------------------
# Emergency keyword catalogue
# ---------------------------------------------------------------------------

# All keywords listed in config/config.yaml (section: agents.emergency_keywords)
# We define them here independently so the test does not depend on YAML loading.
_DOCUMENTED_KEYWORDS = [
    "chest pain",
    "heart attack",
    "stroke",
    "can't breathe",
    "cannot breathe",
    "difficulty breathing",
    "severe bleeding",
    "loss of consciousness",
    "unconscious",
    "overdose",
    "poisoning",
    "suicide",
    "seizure",
    "anaphylaxis",
]


class TestEmergencyKeywordCoverage:
    """Verify every documented keyword triggers emergency detection."""

    @pytest.mark.parametrize("keyword", _DOCUMENTED_KEYWORDS)
    def test_keyword_detected(self, keyword: str):
        from src.agents.supervisor import has_emergency_keywords
        assert has_emergency_keywords(keyword) is True, (
            f"Keyword '{keyword}' should trigger emergency detection."
        )

    @pytest.mark.parametrize("keyword", _DOCUMENTED_KEYWORDS)
    def test_keyword_case_insensitive(self, keyword: str):
        from src.agents.supervisor import has_emergency_keywords
        assert has_emergency_keywords(keyword.upper()) is True
        assert has_emergency_keywords(keyword.title()) is True

    def test_keyword_in_longer_sentence(self):
        """Keywords embedded in a sentence should still be detected."""
        from src.agents.supervisor import has_emergency_keywords
        assert has_emergency_keywords("My friend just had a stroke and fell down.") is True
        assert has_emergency_keywords("She is experiencing severe bleeding from her arm.") is True

    def test_non_emergency_not_triggered(self):
        """Routine medical queries must not trigger emergency."""
        from src.agents.supervisor import has_emergency_keywords
        non_emergency = [
            "What is the normal blood pressure range?",
            "How do I manage Type 2 Diabetes?",
            "What are common side effects of aspirin?",
            "I feel a little tired this afternoon.",
            "Can you recommend a good diet plan?",
        ]
        for text in non_emergency:
            assert has_emergency_keywords(text) is False, (
                f"Non-emergency text '{text}' should NOT trigger detection."
            )


# ---------------------------------------------------------------------------
# emergency_node
# ---------------------------------------------------------------------------

class TestEmergencyNode:
    """Tests for the emergency_node LangGraph node."""

    def _make_state(self, message: str = "emergency") -> dict:
        return {
            "messages": [HumanMessage(content=message)],
            "session_id": "test-emergency",
            "urgency_level": 5,
            "is_emergency": True,
        }

    def test_returns_non_empty_response(self):
        from src.agents.graph import emergency_node
        result = emergency_node(self._make_state())
        assert isinstance(result.get("agent_response"), str)
        assert len(result["agent_response"]) > 0

    def test_emergency_flag_remains_true(self):
        from src.agents.graph import emergency_node
        result = emergency_node(self._make_state())
        assert result.get("is_emergency") is True

    def test_urgency_level_is_five(self):
        from src.agents.graph import emergency_node
        result = emergency_node(self._make_state())
        assert result.get("urgency_level") == 5

    def test_response_contains_call_instruction(self):
        """Emergency response must instruct user to call emergency services."""
        from src.agents.graph import emergency_node
        result = emergency_node(self._make_state())
        response_lower = result["agent_response"].lower()
        # At minimum one of these should appear
        assert any(
            kw in response_lower for kw in ["call", "emergency", "911", "115"]
        ), "Emergency response must instruct user to call for help."

    def test_preserves_session_id(self):
        from src.agents.graph import emergency_node
        state = self._make_state()
        state["session_id"] = "preserve-me"
        result = emergency_node(state)
        assert result.get("session_id") == "preserve-me"

    def test_sources_is_list(self):
        from src.agents.graph import emergency_node
        result = emergency_node(self._make_state())
        assert isinstance(result.get("sources"), list)


# ---------------------------------------------------------------------------
# End-to-end: supervisor routes emergency keywords to emergency_agent
# ---------------------------------------------------------------------------

class TestEmergencyRoutingIntegration:
    """Integration tests: keyword detection → correct routing in supervisor."""

    @pytest.mark.parametrize("query", [
        "I have chest pain",
        "He cannot breathe",
        "This looks like a stroke",
        "Patient is unconscious",
        "Overdose suspected",
    ])
    def test_emergency_queries_route_to_emergency_agent(self, query: str):
        """All emergency queries must route to emergency_agent without LLM call."""
        from src.agents.supervisor import supervisor_node

        # No LLM mock — keyword bypass must fire before LLM
        state = {"messages": [HumanMessage(content=query)]}
        result = supervisor_node(state)

        assert result["route_to"] == "emergency_agent", (
            f"Query '{query}' should route to emergency_agent, got '{result['route_to']}'."
        )
        assert result["is_emergency"] is True
        assert result["urgency_level"] == 5
