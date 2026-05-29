"""
tests/test_tools.py

Unit tests for src/agents/tools.py

Tests cover:
- Input validation (empty / whitespace inputs)
- Return-type contracts (always str)
- Error handling (tool functions never raise)
- Knowledge-base stub substitution
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_doc(text: str, source: str = "TestSource"):
    """Return a minimal Document-like object."""
    doc = MagicMock()
    doc.page_content = text
    doc.metadata = {"source": source}
    return doc


# ---------------------------------------------------------------------------
# search_knowledge_base
# ---------------------------------------------------------------------------

class TestSearchKnowledgeBase:
    """Tests for the search_knowledge_base tool."""

    def test_returns_string(self):
        """Return type must always be str."""
        with patch("src.agents.tools._get_retriever") as mock_retriever_fn:
            mock_retriever = MagicMock()
            mock_retriever.invoke.return_value = [
                _make_fake_doc("Diabetes is a metabolic disease.")
            ]
            mock_retriever_fn.return_value = mock_retriever

            from src.agents.tools import search_knowledge_base
            result = search_knowledge_base.invoke("diabetes")

        assert isinstance(result, str)

    def test_empty_query_returns_message(self):
        """Empty query should return a helpful message, not raise."""
        from src.agents.tools import search_knowledge_base
        result = search_knowledge_base.invoke("")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_whitespace_query_returns_message(self):
        """Whitespace-only query should be handled gracefully."""
        from src.agents.tools import search_knowledge_base
        result = search_knowledge_base.invoke("   ")
        assert isinstance(result, str)

    def test_retriever_error_returns_fallback(self):
        """If the retriever raises, tool returns an error string (does not raise)."""
        with patch("src.agents.tools._get_retriever") as mock_retriever_fn:
            mock_retriever = MagicMock()
            mock_retriever.invoke.side_effect = RuntimeError("DB unavailable")
            mock_retriever_fn.return_value = mock_retriever

            from src.agents.tools import search_knowledge_base
            result = search_knowledge_base.invoke("diabetes")

        assert isinstance(result, str)
        assert "unavailable" in result.lower() or "error" in result.lower()

    def test_no_docs_returns_message(self):
        """When retriever returns empty list, return a 'no documents' message."""
        with patch("src.agents.tools._get_retriever") as mock_retriever_fn:
            mock_retriever = MagicMock()
            mock_retriever.invoke.return_value = []
            mock_retriever_fn.return_value = mock_retriever

            from src.agents.tools import search_knowledge_base
            result = search_knowledge_base.invoke("xyzzy medical")

        assert isinstance(result, str)
        assert "no relevant" in result.lower()


# ---------------------------------------------------------------------------
# assess_urgency
# ---------------------------------------------------------------------------

class TestAssessUrgency:
    """Tests for the assess_urgency tool."""

    def test_returns_json_string(self):
        """Response must be a valid JSON string with required keys."""
        import json

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"urgency_level": 3, "is_emergency": false, '
                    '"reasoning": "Moderate symptoms.", "recommended_action": "See a doctor."}'
        )

        with patch("src.agents.tools.get_llm", return_value=mock_llm):
            from src.agents.tools import assess_urgency
            result = assess_urgency.invoke("I have a headache and fever.")

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "urgency_level" in parsed
        assert "is_emergency" in parsed

    def test_empty_symptoms_returns_message(self):
        """Empty symptoms should return a message, not raise."""
        from src.agents.tools import assess_urgency
        result = assess_urgency.invoke("")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_llm_failure_returns_fallback(self):
        """LLM failure must return a safe fallback JSON, not raise."""
        import json

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API error")

        with patch("src.agents.tools.get_llm", return_value=mock_llm):
            from src.agents.tools import assess_urgency
            result = assess_urgency.invoke("I feel dizzy.")

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "urgency_level" in parsed

    def test_urgency_level_in_range(self):
        """Urgency level must be 1-5."""
        import json

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"urgency_level": 2, "is_emergency": false, '
                    '"reasoning": "Minor symptoms.", "recommended_action": "Rest."}'
        )

        with patch("src.agents.tools.get_llm", return_value=mock_llm):
            from src.agents.tools import assess_urgency
            result = assess_urgency.invoke("I have a mild headache.")

        parsed = json.loads(result)
        assert 1 <= parsed["urgency_level"] <= 5


# ---------------------------------------------------------------------------
# explain_medical_term
# ---------------------------------------------------------------------------

class TestExplainMedicalTerm:
    """Tests for the explain_medical_term tool."""

    def test_returns_string(self):
        """Return must always be str."""
        with patch("src.agents.tools._get_retriever") as mock_retriever_fn:
            mock_retriever = MagicMock()
            mock_retriever.invoke.return_value = [
                _make_fake_doc("Hypertension is high blood pressure.")
            ]
            mock_retriever_fn.return_value = mock_retriever

            from src.agents.tools import explain_medical_term
            result = explain_medical_term.invoke("hypertension")

        assert isinstance(result, str)

    def test_empty_term_returns_message(self):
        from src.agents.tools import explain_medical_term
        result = explain_medical_term.invoke("")
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# get_drug_info
# ---------------------------------------------------------------------------

class TestGetDrugInfo:
    """Tests for the get_drug_info tool."""

    def test_returns_string(self):
        with patch("src.agents.tools._get_retriever") as mock_retriever_fn:
            mock_retriever = MagicMock()
            mock_retriever.invoke.return_value = [
                _make_fake_doc("Metformin is used to treat Type 2 Diabetes.")
            ]
            mock_retriever_fn.return_value = mock_retriever

            from src.agents.tools import get_drug_info
            result = get_drug_info.invoke("metformin")

        assert isinstance(result, str)

    def test_empty_drug_name_returns_message(self):
        from src.agents.tools import get_drug_info
        result = get_drug_info.invoke("")
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# get_lifestyle_advice
# ---------------------------------------------------------------------------

class TestGetLifestyleAdvice:
    """Tests for the get_lifestyle_advice tool."""

    def test_returns_string(self):
        with patch("src.agents.tools._get_retriever") as mock_retriever_fn:
            mock_retriever = MagicMock()
            mock_retriever.invoke.return_value = [
                _make_fake_doc("Exercise 30 minutes a day to manage diabetes.")
            ]
            mock_retriever_fn.return_value = mock_retriever

            from src.agents.tools import get_lifestyle_advice
            result = get_lifestyle_advice.invoke("diabetes management")

        assert isinstance(result, str)

    def test_empty_condition_returns_message(self):
        from src.agents.tools import get_lifestyle_advice
        result = get_lifestyle_advice.invoke("")
        assert isinstance(result, str)
        assert len(result) > 0
