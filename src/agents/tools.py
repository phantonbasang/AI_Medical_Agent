"""
src/agents/tools.py

LangChain tool definitions for the AI Medical Agent.

Each tool is decorated with ``@tool`` so it can be bound to a ReAct
agent via ``llm.bind_tools([...])``.  All tools include:
  - A clear docstring (used by the LLM to decide when to call the tool)
  - Input validation
  - Graceful error handling (never raises — returns an error string)
  - Retry logic via ``tenacity`` for transient failures

Tools
-----
- search_knowledge_base   : Semantic search over ChromaDB
- assess_urgency          : Rate symptom severity 1-5
- explain_medical_term    : Plain-language definitions
- get_drug_info           : Drug / medication information
- get_lifestyle_advice    : Prevention and wellness tips
"""
from __future__ import annotations

import json
import re

from langchain_core.tools import tool
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.utils import get_logger, safe_extract_content

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy retriever — loaded on first tool call to avoid import-time side effects
# ---------------------------------------------------------------------------
_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from src.rag.vectorstore import get_retriever  # local import avoids circular
        _retriever = get_retriever()
    return _retriever


def _format_docs(docs) -> str:
    """Concatenate retrieved document texts with a separator."""
    if not docs:
        return "No relevant documents found in the knowledge base."
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', 'Unknown')}]\n{d.page_content}"
        for d in docs
    )


# ---------------------------------------------------------------------------
# Tool: Knowledge base search
# ---------------------------------------------------------------------------

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the medical knowledge base (PubMedQA + MedQuAD) for information
    relevant to the query.  Use this for general medical questions, disease
    explanations, symptoms, treatments, and evidence-based information.

    Parameters
    ----------
    query:
        The medical question or topic to search for.

    Returns
    -------
    str
        Relevant excerpts from the knowledge base, or an error message if
        the search fails.
    """
    if not query or not query.strip():
        return "Please provide a non-empty search query."

    try:
        retriever = _get_retriever()
        docs = retriever.invoke(query.strip())
        result = _format_docs(docs)
        logger.debug("search_knowledge_base('%s'): %d docs returned", query[:60], len(docs))
        return result
    except Exception as exc:
        logger.warning("search_knowledge_base failed: %s", exc)
        return (
            f"Knowledge base search unavailable ({type(exc).__name__}). "
            "Providing answer from built-in medical knowledge."
        )


# ---------------------------------------------------------------------------
# Tool: Urgency assessment
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type(Exception),
    reraise=False,
)
def _call_llm_for_urgency(symptoms: str) -> dict:
    """Inner call so tenacity can retry without re-importing the LLM."""
    from src.models.llm import get_llm  # local import

    llm = get_llm()
    prompt = f"""You are a medical triage assistant. Assess the urgency of these symptoms.

Symptoms: {symptoms}

Respond ONLY with valid JSON in exactly this format (no extra text):
{{
  "urgency_level": <integer 1-5>,
  "is_emergency": <true|false>,
  "reasoning": "<one sentence>",
  "recommended_action": "<brief advice>"
}}

Urgency scale:
1 = Routine (no immediate concern)
2 = Low (schedule an appointment)
3 = Moderate (see a doctor soon)
4 = High (seek urgent care today)
5 = Critical (call emergency services immediately)"""

    response = llm.invoke(prompt)
    raw = safe_extract_content(response)

    # Extract JSON from the response
    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in LLM response: {raw[:200]}")

    return json.loads(match.group())


@tool
def assess_urgency(symptoms: str) -> str:
    """
    Assess the medical urgency of described symptoms on a 1-5 scale.
    Use this tool when a user describes symptoms that need triage.

    Parameters
    ----------
    symptoms:
        A description of the patient's symptoms.

    Returns
    -------
    str
        JSON-formatted urgency assessment with level, emergency flag,
        reasoning, and recommended action.
    """
    if not symptoms or not symptoms.strip():
        return "Please describe the symptoms to assess."

    try:
        result = _call_llm_for_urgency(symptoms)
        # Normalise types
        result["urgency_level"] = int(result.get("urgency_level", 3))
        result["is_emergency"] = bool(result.get("is_emergency", False))
        logger.info(
            "assess_urgency: level=%d  emergency=%s",
            result["urgency_level"],
            result["is_emergency"],
        )
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("assess_urgency failed: %s", exc)
        return json.dumps(
            {
                "urgency_level": 3,
                "is_emergency": False,
                "reasoning": "Unable to assess automatically — please consult a healthcare professional.",
                "recommended_action": "Contact your doctor or a telehealth service.",
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# Tool: Medical term explanation
# ---------------------------------------------------------------------------

@tool
def explain_medical_term(term: str) -> str:
    """
    Provide a clear, plain-language explanation of a medical term,
    condition, or concept.  Use this when the user asks 'what is X?'
    or needs a definition.

    Parameters
    ----------
    term:
        The medical term, condition name, or concept to explain.

    Returns
    -------
    str
        A plain-language explanation suitable for a non-specialist.
    """
    if not term or not term.strip():
        return "Please provide a medical term to explain."

    # First try the knowledge base
    kb_result = search_knowledge_base.invoke(f"definition explanation {term}")  # type: ignore[attr-defined]
    if "No relevant documents" not in kb_result:
        return kb_result

    # Fallback to LLM
    try:
        from src.models.llm import get_llm
        llm = get_llm()
        prompt = (
            f"Explain the medical term '{term}' in simple language that a "
            "patient with no medical background can understand. "
            "Include: definition, what causes it (if applicable), and key facts."
        )
        return safe_extract_content(llm.invoke(prompt))
    except Exception as exc:
        logger.warning("explain_medical_term LLM fallback failed: %s", exc)
        return f"Unable to retrieve explanation for '{term}' at this time."


# ---------------------------------------------------------------------------
# Tool: Drug information
# ---------------------------------------------------------------------------

@tool
def get_drug_info(drug_name: str) -> str:
    """
    Retrieve evidence-based information about a medication, including its
    uses, common side effects, dosage guidelines, and contraindications.
    Use this when the user asks about a specific drug or medication.

    Parameters
    ----------
    drug_name:
        Name of the drug or medication (generic or brand name).

    Returns
    -------
    str
        Drug information from the knowledge base or built-in LLM knowledge.
    """
    if not drug_name or not drug_name.strip():
        return "Please provide a drug or medication name."

    query = f"medication drug {drug_name} uses side effects dosage"
    kb_result = search_knowledge_base.invoke(query)  # type: ignore[attr-defined]
    if "No relevant documents" not in kb_result:
        return kb_result

    try:
        from src.models.llm import get_llm
        llm = get_llm()
        prompt = (
            f"Provide evidence-based information about the medication '{drug_name}'. "
            "Include: primary uses, mechanism of action (briefly), common side effects, "
            "and important warnings. Use simple language."
        )
        return safe_extract_content(llm.invoke(prompt))
    except Exception as exc:
        logger.warning("get_drug_info LLM fallback failed: %s", exc)
        return f"Unable to retrieve drug information for '{drug_name}' at this time."


# ---------------------------------------------------------------------------
# Tool: Lifestyle advice
# ---------------------------------------------------------------------------

@tool
def get_lifestyle_advice(condition_or_goal: str) -> str:
    """
    Provide personalised lifestyle and prevention advice related to a
    medical condition or health goal.  Use this for questions about diet,
    exercise, sleep, stress management, and general wellness.

    Parameters
    ----------
    condition_or_goal:
        A medical condition (e.g. 'hypertension') or health goal
        (e.g. 'lose weight', 'improve sleep').

    Returns
    -------
    str
        Evidence-based lifestyle recommendations.
    """
    if not condition_or_goal or not condition_or_goal.strip():
        return "Please describe the condition or health goal."

    query = f"lifestyle prevention diet exercise {condition_or_goal}"
    kb_result = search_knowledge_base.invoke(query)  # type: ignore[attr-defined]
    if "No relevant documents" not in kb_result:
        return kb_result

    try:
        from src.models.llm import get_llm
        llm = get_llm()
        prompt = (
            f"Provide practical, evidence-based lifestyle advice for someone dealing with "
            f"'{condition_or_goal}'. Cover: diet recommendations, physical activity, "
            "sleep hygiene, stress management, and when to seek professional help."
        )
        return safe_extract_content(llm.invoke(prompt))
    except Exception as exc:
        logger.warning("get_lifestyle_advice LLM fallback failed: %s", exc)
        return f"Unable to retrieve lifestyle advice for '{condition_or_goal}' at this time."


# ---------------------------------------------------------------------------
# Tool registry helpers
# ---------------------------------------------------------------------------

# Grouped tool lists for each agent
RAG_TOOLS = [search_knowledge_base, explain_medical_term]
SYMPTOM_TOOLS = [assess_urgency, search_knowledge_base]
DRUG_TOOLS = [get_drug_info, search_knowledge_base]
LIFESTYLE_TOOLS = [get_lifestyle_advice, search_knowledge_base]
ALL_TOOLS = [
    search_knowledge_base,
    assess_urgency,
    explain_medical_term,
    get_drug_info,
    get_lifestyle_advice,
]
