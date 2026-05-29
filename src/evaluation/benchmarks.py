"""
src/evaluation/benchmarks.py

Deterministic benchmark scenarios for the AI Medical Agent.

Benchmark categories
--------------------
1. Routing accuracy  : Does the supervisor route to the correct agent?
2. Emergency bypass  : Does the agent immediately respond to emergencies?
3. Tool invocation   : Do tools return non-empty, sensible responses?
4. Response quality  : Heuristic checks on response length and content.

Usage
-----
    from src.evaluation.benchmarks import run_benchmarks
    results = run_benchmarks(graph)
    print(results.to_string())
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from src.agents.supervisor import has_emergency_keywords
from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Test case definition
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkCase:
    """Single benchmark test case."""
    case_id: str
    description: str
    query: str
    expected_route: str                  # Expected agent node name
    expected_emergency: bool = False
    min_response_length: int = 50        # Minimum char count in response
    required_keywords: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Built-in benchmark suite
# ---------------------------------------------------------------------------

BENCHMARK_CASES: list[BenchmarkCase] = [
    # --- General Q&A ---
    BenchmarkCase(
        case_id="rag_01",
        description="General disease question → RAG agent",
        query="What is Type 2 Diabetes and how does it develop?",
        expected_route="rag_agent",
        min_response_length=100,
        required_keywords=["diabetes", "insulin"],
    ),
    BenchmarkCase(
        case_id="rag_02",
        description="Medical term definition → RAG agent",
        query="Can you explain what hypertension means?",
        expected_route="rag_agent",
        min_response_length=80,
        required_keywords=["blood pressure"],
    ),
    # --- Symptom / Triage ---
    BenchmarkCase(
        case_id="symptom_01",
        description="Symptom description → Symptom agent",
        query="I've been having persistent headaches for 5 days. Should I be worried?",
        expected_route="symptom_agent",
        min_response_length=100,
    ),
    BenchmarkCase(
        case_id="symptom_02",
        description="Multiple vague symptoms → Symptom agent",
        query="I feel tired all the time, have frequent thirst, and need to urinate often.",
        expected_route="symptom_agent",
        min_response_length=100,
    ),
    # --- Drug / Medication ---
    BenchmarkCase(
        case_id="drug_01",
        description="Medication info → Drug agent",
        query="What are the side effects of metformin?",
        expected_route="drug_agent",
        min_response_length=80,
        required_keywords=["metformin"],
    ),
    BenchmarkCase(
        case_id="drug_02",
        description="Drug interaction question → Drug agent",
        query="Can I take ibuprofen and paracetamol together?",
        expected_route="drug_agent",
        min_response_length=80,
    ),
    # --- Lifestyle ---
    BenchmarkCase(
        case_id="lifestyle_01",
        description="Diet question → Lifestyle agent",
        query="What foods should I avoid if I have high blood pressure?",
        expected_route="lifestyle_agent",
        min_response_length=100,
    ),
    BenchmarkCase(
        case_id="lifestyle_02",
        description="Exercise advice → Lifestyle agent",
        query="What type of exercise is best for managing Type 2 Diabetes?",
        expected_route="lifestyle_agent",
        min_response_length=100,
    ),
    # --- Emergency ---
    BenchmarkCase(
        case_id="emergency_01",
        description="Chest pain → Emergency bypass",
        query="I have severe chest pain radiating to my left arm and difficulty breathing.",
        expected_route="emergency_agent",
        expected_emergency=True,
        min_response_length=50,
        required_keywords=["emergency", "call"],
    ),
    BenchmarkCase(
        case_id="emergency_02",
        description="Stroke symptoms → Emergency bypass",
        query="My father is having a stroke right now, one side of his face is drooping!",
        expected_route="emergency_agent",
        expected_emergency=True,
        min_response_length=50,
        required_keywords=["emergency", "call"],
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Result for a single benchmark case."""
    case_id: str
    description: str
    query: str
    expected_route: str
    actual_route: str
    expected_emergency: bool
    actual_emergency: bool
    route_correct: bool
    emergency_correct: bool
    response_length: int
    length_ok: bool
    keywords_ok: bool
    latency_ms: float
    passed: bool
    response_preview: str


def run_benchmarks(
    graph,
    cases: list[BenchmarkCase] | None = None,
    session_prefix: str = "benchmark",
) -> pd.DataFrame:
    """
    Run all benchmark cases against the compiled medical graph.

    Parameters
    ----------
    graph:
        Compiled LangGraph returned by ``build_medical_graph()``.
    cases:
        List of ``BenchmarkCase`` objects.  Defaults to ``BENCHMARK_CASES``.
    session_prefix:
        Prefix for LangGraph thread IDs.

    Returns
    -------
    pd.DataFrame
        One row per benchmark case with pass/fail columns and latency.
    """
    from langchain_core.messages import HumanMessage

    if cases is None:
        cases = BENCHMARK_CASES

    results: list[BenchmarkResult] = []

    for case in cases:
        logger.info("[benchmark] Running case '%s': %s", case.case_id, case.description)
        config = {"configurable": {"thread_id": f"{session_prefix}-{case.case_id}"}}
        start_ms = time.time() * 1000

        try:
            state = graph.invoke(
                {"messages": [HumanMessage(content=case.query)]},
                config=config,
            )
            latency_ms = time.time() * 1000 - start_ms

            actual_route: str = state.get("route_to", "unknown")
            actual_emergency: bool = bool(state.get("is_emergency", False))
            response: str = state.get("agent_response", "")

        except Exception as exc:
            latency_ms = time.time() * 1000 - start_ms
            logger.error("[benchmark] Case '%s' raised: %s", case.case_id, exc)
            actual_route = "error"
            actual_emergency = False
            response = ""

        route_correct = actual_route == case.expected_route
        emergency_correct = actual_emergency == case.expected_emergency
        response_length = len(response)
        length_ok = response_length >= case.min_response_length
        keywords_ok = all(kw.lower() in response.lower() for kw in case.required_keywords)
        passed = all([route_correct, emergency_correct, length_ok, keywords_ok])

        result = BenchmarkResult(
            case_id=case.case_id,
            description=case.description,
            query=case.query[:60] + "…" if len(case.query) > 60 else case.query,
            expected_route=case.expected_route,
            actual_route=actual_route,
            expected_emergency=case.expected_emergency,
            actual_emergency=actual_emergency,
            route_correct=route_correct,
            emergency_correct=emergency_correct,
            response_length=response_length,
            length_ok=length_ok,
            keywords_ok=keywords_ok,
            latency_ms=round(latency_ms, 1),
            passed=passed,
            response_preview=response[:120],
        )
        results.append(result)

        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(
            "[benchmark] %s %s — route_ok=%s  emergency_ok=%s  latency=%.0f ms",
            status,
            case.case_id,
            route_correct,
            emergency_correct,
            latency_ms,
        )

    df = pd.DataFrame([vars(r) for r in results])

    # Summary
    n_pass = df["passed"].sum()
    n_total = len(df)
    logger.info(
        "Benchmark complete: %d/%d passed (%.1f%%)",
        n_pass,
        n_total,
        100.0 * n_pass / n_total if n_total else 0,
    )
    return df


def emergency_keyword_benchmark() -> pd.DataFrame:
    """
    Lightweight benchmark for the emergency keyword detector (no LLM needed).

    Returns
    -------
    pd.DataFrame
        Pass/fail results for each test case.
    """
    cases = [
        # True positives
        ("chest pain severe", True),
        ("I cannot breathe", True),
        ("overdose on pills", True),
        ("stroke symptoms now", True),
        ("patient is unconscious", True),
        # True negatives
        ("headache for 2 days", False),
        ("what is diabetes", False),
        ("metformin side effects", False),
        ("diet for hypertension", False),
        ("I feel tired", False),
    ]

    rows = []
    for text, expected in cases:
        actual = has_emergency_keywords(text)
        rows.append(
            {
                "query": text,
                "expected_emergency": expected,
                "actual_emergency": actual,
                "passed": expected == actual,
            }
        )

    df = pd.DataFrame(rows)
    n_pass = df["passed"].sum()
    logger.info(
        "Emergency keyword benchmark: %d/%d passed", n_pass, len(df)
    )
    return df
