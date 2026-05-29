"""
src/rag/loader.py

Dataset loaders for the medical knowledge base.

Supports:
  - PubMedQA  (qiaojin/PubMedQA  on HuggingFace)
  - MedQuAD   (keivalya/MedQuad-MedicalQnADataset on HuggingFace)

Each loader returns a list of ``langchain_core.documents.Document`` objects
ready to be fed into ``preprocess_documents()``.

Usage
-----
    from src.rag.loader import load_pubmedqa, load_medquad
    docs = load_pubmedqa(limit=1000) + load_medquad(limit=1000)
"""
from __future__ import annotations

from langchain_core.documents import Document

from src.utils import get_logger, load_config

logger = get_logger(__name__)


def load_pubmedqa(limit: int | None = None) -> list[Document]:
    """
    Load samples from the PubMedQA dataset (HuggingFace).

    Each sample is converted to a ``Document`` whose page content
    combines the question, the long answer and (when present) the
    final yes/no/maybe answer.

    Parameters
    ----------
    limit:
        Maximum number of samples to load.  ``None`` uses the value
        from ``config.yaml`` (``data.pubmedqa_limit``).

    Returns
    -------
    list[Document]
        List of LangChain Document objects with metadata.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "datasets package is required. Run: pip install datasets"
        ) from exc

    cfg = load_config()
    if limit is None:
        limit = int(cfg.get("data", {}).get("pubmedqa_limit", 1000))

    logger.info("Loading PubMedQA (limit=%d)…", limit)

    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    docs: list[Document] = []

    for item in dataset.select(range(min(limit, len(dataset)))):
        question: str = item.get("question", "")
        # long_answer is the main prose answer
        long_answer: str = item.get("long_answer", "")
        final_decision: str = item.get("final_decision", "")
        pubid: str = str(item.get("pubid", ""))

        content_parts = [f"Question: {question}"]
        if long_answer:
            content_parts.append(f"Answer: {long_answer}")
        if final_decision:
            content_parts.append(f"Conclusion: {final_decision}")

        docs.append(
            Document(
                page_content="\n".join(content_parts),
                metadata={
                    "source": "PubMedQA",
                    "pubid": pubid,
                    "final_decision": final_decision,
                },
            )
        )

    logger.info("Loaded %d PubMedQA documents.", len(docs))
    return docs


def load_medquad(limit: int | None = None) -> list[Document]:
    """
    Load samples from the MedQuAD dataset (HuggingFace).

    MedQuAD is a collection of medical Q&A pairs sourced from NIH
    websites covering diseases, drugs and treatments.

    Parameters
    ----------
    limit:
        Maximum number of samples to load.  ``None`` uses the value
        from ``config.yaml`` (``data.medquad_limit``).

    Returns
    -------
    list[Document]
        List of LangChain Document objects with metadata.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "datasets package is required. Run: pip install datasets"
        ) from exc

    cfg = load_config()
    if limit is None:
        limit = int(cfg.get("data", {}).get("medquad_limit", 1000))

    logger.info("Loading MedQuAD (limit=%d)…", limit)

    dataset = load_dataset(
        "keivalya/MedQuad-MedicalQnADataset", split="train"
    )
    docs: list[Document] = []

    for item in dataset.select(range(min(limit, len(dataset)))):
        question: str = item.get("Question", "")
        answer: str = item.get("Answer", "")

        if not question or not answer:
            continue

        docs.append(
            Document(
                page_content=f"Question: {question}\nAnswer: {answer}",
                metadata={
                    "source": "MedQuAD",
                    "question_type": item.get("qtype", "general"),
                },
            )
        )

    logger.info("Loaded %d MedQuAD documents.", len(docs))
    return docs
