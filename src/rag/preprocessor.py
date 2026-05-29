"""
src/rag/preprocessor.py

Text cleaning and document chunking for medical documents.

Pipeline
--------
1. ``clean_text()``          — Unicode normalisation, whitespace, special chars
2. ``preprocess_documents()`` — Split raw Documents into fixed-size chunks

Usage
-----
    from src.rag.preprocessor import preprocess_documents
    raw_docs = load_pubmedqa() + load_medquad()
    chunks = preprocess_documents(raw_docs)
"""
from __future__ import annotations

import re
import unicodedata

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils import get_logger, load_config

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Normalise a raw medical text string.

    Steps applied
    -------------
    1. Unicode NFKC normalisation (handles special quotes, ligatures, etc.)
    2. Remove non-printable control characters (keep \\n and \\t)
    3. Collapse multiple whitespace runs to a single space
    4. Strip leading / trailing whitespace

    Parameters
    ----------
    text:
        Raw string from a dataset field.

    Returns
    -------
    str
        Cleaned, normalised string.
    """
    if not text:
        return ""

    # 1. Unicode normalisation
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove non-printable control chars (keep \\n \\t)
    text = re.sub(r"[^\S\n\t]+", " ", text)  # multi-space → single
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 3. Collapse blank lines > 2 consecutive
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Document chunking
# ---------------------------------------------------------------------------

def preprocess_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Clean and split a list of raw Documents into fixed-size chunks.

    Each output chunk inherits the metadata from its parent document and
    receives an additional ``chunk_index`` field for traceability.

    Parameters
    ----------
    documents:
        Raw ``Document`` objects from any loader.
    chunk_size:
        Target token / character count per chunk.  Defaults to
        ``config.yaml`` → ``rag.chunk_size``.
    chunk_overlap:
        Overlap between consecutive chunks.  Defaults to
        ``config.yaml`` → ``rag.chunk_overlap``.

    Returns
    -------
    list[Document]
        Chunked, cleaned Documents ready for embedding.
    """
    cfg = load_config()
    rag_cfg = cfg.get("rag", {})

    if chunk_size is None:
        chunk_size = int(rag_cfg.get("chunk_size", 900))
    if chunk_overlap is None:
        chunk_overlap = int(rag_cfg.get("chunk_overlap", 175))

    logger.info(
        "Preprocessing %d documents (chunk_size=%d, overlap=%d)…",
        len(documents),
        chunk_size,
        chunk_overlap,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    cleaned: list[Document] = []
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
        if doc.page_content:
            cleaned.append(doc)

    chunks: list[Document] = splitter.split_documents(cleaned)

    # Add chunk_index to metadata for later traceability
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx

    logger.info("Produced %d chunks from %d documents.", len(chunks), len(cleaned))
    return chunks
