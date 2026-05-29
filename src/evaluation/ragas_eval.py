"""
src/evaluation/ragas_eval.py

RAGAS-based evaluation of the AI Medical Agent.

Metrics computed
----------------
- answer_relevancy   : How relevant is the answer to the question?
- faithfulness       : Is the answer grounded in the retrieved context?
- context_precision  : Are the retrieved chunks actually useful?
- context_recall     : Did we retrieve all the needed information?

Usage
-----
    from src.evaluation.ragas_eval import evaluate_responses

    results = evaluate_responses([
        {
            "question": "What is diabetes?",
            "answer": "Diabetes is ...",
            "contexts": ["...retrieved chunk 1...", "...chunk 2..."],
            "ground_truth": "Diabetes mellitus is a metabolic disease...",  # optional
        }
    ])
    print(results.to_pandas())
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from src.utils import get_logger, load_config

logger = get_logger(__name__)


def evaluate_responses(
    samples: list[dict],
    output_file: str | None = None,
) -> pd.DataFrame:
    """
    Run RAGAS evaluation on a list of QA samples.

    Parameters
    ----------
    samples:
        List of dicts with keys:
          - ``question``    (str)
          - ``answer``      (str)
          - ``contexts``    (list[str])
          - ``ground_truth``(str, optional — required for context_recall)
    output_file:
        If given, save the results DataFrame as CSV to this path.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per sample and one column per metric.
    """
    if not samples:
        raise ValueError("No samples provided for evaluation.")

    # Validate required keys
    for i, s in enumerate(samples):
        for key in ("question", "answer", "contexts"):
            if key not in s:
                raise ValueError(f"Sample {i} is missing required key '{key}'.")
        if not isinstance(s["contexts"], list):
            raise TypeError(f"Sample {i}: 'contexts' must be a list of strings.")

    try:
        from datasets import Dataset  # type: ignore[import]
        from ragas import evaluate  # type: ignore[import]
        from ragas.metrics import (  # type: ignore[import]
            answer_relevancy,
            context_precision,
            faithfulness,
        )
    except ImportError as exc:
        raise ImportError(
            "RAGAS and/or datasets not installed. "
            "Run: pip install ragas datasets"
        ) from exc

    cfg = load_config()
    metrics_cfg: list[str] = cfg.get("evaluation", {}).get(
        "ragas_metrics",
        ["answer_relevancy", "faithfulness", "context_precision"],
    )

    # Build metric objects (skip context_recall if no ground_truth)
    metric_map = {
        "answer_relevancy": answer_relevancy,
        "faithfulness": faithfulness,
        "context_precision": context_precision,
    }
    # Try to import context_recall — available in newer ragas versions
    try:
        from ragas.metrics import context_recall  # type: ignore[import]
        metric_map["context_recall"] = context_recall
    except ImportError:
        logger.debug("context_recall metric not available in this ragas version.")

    selected_metrics = [metric_map[m] for m in metrics_cfg if m in metric_map]

    if not selected_metrics:
        raise ValueError(f"No valid RAGAS metrics found in config: {metrics_cfg}")

    # Build HuggingFace Dataset
    data_dict: dict[str, list] = {
        "question": [s["question"] for s in samples],
        "answer": [s["answer"] for s in samples],
        "contexts": [s["contexts"] for s in samples],
    }
    has_ground_truth = all("ground_truth" in s for s in samples)
    if has_ground_truth:
        data_dict["ground_truth"] = [s["ground_truth"] for s in samples]

    dataset = Dataset.from_dict(data_dict)

    logger.info(
        "Running RAGAS evaluation on %d samples with metrics: %s",
        len(samples),
        [m.name for m in selected_metrics],
    )

    result = evaluate(dataset, metrics=selected_metrics)
    df: pd.DataFrame = result.to_pandas()

    logger.info("RAGAS evaluation complete.")
    logger.info("Mean scores:\n%s", df.mean(numeric_only=True).to_string())

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.info("Results saved to '%s'.", output_file)

    return df


def evaluate_agent(
    graph,
    qa_pairs: list[dict],
    session_prefix: str = "eval",
    output_file: str | None = None,
) -> pd.DataFrame:
    """
    Convenience wrapper: run the agent on each QA pair, then evaluate.

    Parameters
    ----------
    graph:
        Compiled LangGraph (from ``build_medical_graph()``).
    qa_pairs:
        List of dicts with keys ``question`` and (optionally) ``ground_truth``.
    session_prefix:
        Prefix for thread IDs used during evaluation.
    output_file:
        Optional CSV path for saving results.

    Returns
    -------
    pd.DataFrame
        RAGAS evaluation results.
    """
    from langchain_core.messages import HumanMessage
    from src.rag.vectorstore import get_retriever

    retriever = get_retriever()
    samples: list[dict] = []

    for i, pair in enumerate(qa_pairs):
        question = pair["question"]
        logger.info("[eval %d/%d] Question: %s", i + 1, len(qa_pairs), question[:80])

        config = {"configurable": {"thread_id": f"{session_prefix}-{i}"}}
        try:
            result = graph.invoke(
                {"messages": [HumanMessage(content=question)]},
                config=config,
            )
            answer = result.get("agent_response", "")
        except Exception as exc:
            logger.warning("Agent invocation failed for sample %d: %s", i, exc)
            answer = ""

        # Retrieve contexts for faithfulness check
        try:
            docs = retriever.invoke(question)
            contexts = [d.page_content for d in docs]
        except Exception:
            contexts = []

        sample: dict = {
            "question": question,
            "answer": answer,
            "contexts": contexts,
        }
        if "ground_truth" in pair:
            sample["ground_truth"] = pair["ground_truth"]

        samples.append(sample)

    return evaluate_responses(samples, output_file=output_file)
