"""
src/rag/vectorstore.py

ChromaDB vector store management.

Functions
---------
- ``build_vectorstore()``  — Embed chunks and persist to disk (run once)
- ``load_vectorstore()``   — Reconnect to an existing persisted store
- ``get_retriever()``      — Return a LangChain retriever from the store

Usage
-----
    # First time (slow — embeds everything)
    from src.rag.vectorstore import build_vectorstore
    build_vectorstore()

    # Subsequent runs (fast)
    from src.rag.vectorstore import get_retriever
    retriever = get_retriever()
    docs = retriever.invoke("symptoms of diabetes")
"""
from __future__ import annotations

import os
from typing import Any

from langchain_chroma import Chroma  # type: ignore[import]
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from tqdm import tqdm

from src.models.embeddings import get_embeddings
from src.utils import get_logger, load_config

logger = get_logger(__name__)

_vectorstore_instance: Chroma | None = None


def _get_config() -> dict[str, Any]:
    cfg = load_config()
    return cfg.get("rag", {})


def build_vectorstore(
    chunks: list[Document],
    persist_dir: str | None = None,
    collection_name: str | None = None,
    batch_size: int = 500,
) -> Chroma:
    """
    Embed a list of document chunks and persist them in ChromaDB.

    This is the expensive one-time operation.  Progress is shown via
    ``tqdm``.  Subsequent runs should use ``load_vectorstore()`` instead.

    Parameters
    ----------
    chunks:
        Pre-processed ``Document`` chunks (from ``preprocess_documents``).
    persist_dir:
        Directory where ChromaDB will write its files.  Defaults to
        ``config.yaml`` → ``rag.chroma_persist_dir``.
    collection_name:
        ChromaDB collection name.  Defaults to
        ``config.yaml`` → ``rag.collection_name``.
    batch_size:
        Number of chunks to add per batch (reduce if OOM).

    Returns
    -------
    Chroma
        Initialised and persisted ChromaDB instance.
    """
    global _vectorstore_instance

    rag_cfg = _get_config()
    if persist_dir is None:
        persist_dir = os.getenv(
            "CHROMA_PERSIST_DIR", rag_cfg.get("chroma_persist_dir", "./chroma_medical_db")
        )
    if collection_name is None:
        collection_name = rag_cfg.get("collection_name", "medical_knowledge_base")

    embeddings = get_embeddings()
    logger.info(
        "Building vectorstore: %d chunks → %s (collection=%s)",
        len(chunks),
        persist_dir,
        collection_name,
    )

    vectorstore: Chroma | None = None
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks"):
        batch = chunks[i : i + batch_size]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                batch,
                embeddings,
                persist_directory=persist_dir,
                collection_name=collection_name,
            )
        else:
            vectorstore.add_documents(batch)

        logger.debug(
            "Batch %d/%d done (%d docs total)",
            i // batch_size + 1,
            (len(chunks) + batch_size - 1) // batch_size,
            min(i + batch_size, len(chunks)),
        )

    if vectorstore is None:
        raise ValueError("No chunks provided — vectorstore was not created.")

    logger.info("Vectorstore built and persisted at '%s'.", persist_dir)
    _vectorstore_instance = vectorstore
    return vectorstore


def load_vectorstore(
    persist_dir: str | None = None,
    collection_name: str | None = None,
) -> Chroma:
    """
    Load an existing ChromaDB vectorstore from disk.

    Parameters
    ----------
    persist_dir:
        Directory where ChromaDB files are stored.
    collection_name:
        ChromaDB collection name.

    Returns
    -------
    Chroma
        Loaded vectorstore.

    Raises
    ------
    FileNotFoundError
        When *persist_dir* does not exist (vectorstore not built yet).
    """
    global _vectorstore_instance
    if _vectorstore_instance is not None:
        return _vectorstore_instance

    rag_cfg = _get_config()
    if persist_dir is None:
        persist_dir = os.getenv(
            "CHROMA_PERSIST_DIR", rag_cfg.get("chroma_persist_dir", "./chroma_medical_db")
        )
    if collection_name is None:
        collection_name = rag_cfg.get("collection_name", "medical_knowledge_base")

    if not os.path.exists(persist_dir):
        raise FileNotFoundError(
            f"ChromaDB persist directory not found: '{persist_dir}'. "
            "Run build_vectorstore() first."
        )

    embeddings = get_embeddings()
    logger.info("Loading vectorstore from '%s'…", persist_dir)

    _vectorstore_instance = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    count = _vectorstore_instance._collection.count()
    logger.info("Vectorstore loaded: %d documents.", count)
    return _vectorstore_instance


def get_retriever(
    persist_dir: str | None = None,
    top_k: int | None = None,
) -> VectorStoreRetriever:
    """
    Return a LangChain retriever backed by the persisted ChromaDB store.

    Parameters
    ----------
    persist_dir:
        Directory where ChromaDB files are stored.
    top_k:
        Number of documents to retrieve per query.  Defaults to
        ``config.yaml`` → ``rag.top_k``.

    Returns
    -------
    VectorStoreRetriever
    """
    rag_cfg = _get_config()
    if top_k is None:
        top_k = int(rag_cfg.get("top_k", 5))

    vectorstore = load_vectorstore(persist_dir=persist_dir)
    return vectorstore.as_retriever(search_kwargs={"k": top_k})
