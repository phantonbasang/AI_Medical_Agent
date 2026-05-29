"""
src/models/embeddings.py

Centralised embedding model initialisation.

Uses FastEmbedEmbeddings (BAAI/bge-small-en-v1.5 by default) which runs
fully locally — no external API key required.

Usage
-----
    from src.models.embeddings import get_embeddings
    embeddings = get_embeddings()
    vectors = embeddings.embed_documents(["What is diabetes?"])
"""
from __future__ import annotations

from langchain_community.embeddings import FastEmbedEmbeddings  # type: ignore[import]

from src.utils import get_logger, load_config

logger = get_logger(__name__)

_embeddings_instance: FastEmbedEmbeddings | None = None


def get_embeddings(force_new: bool = False) -> FastEmbedEmbeddings:
    """
    Return a cached FastEmbedEmbeddings instance.

    Model name and cache folder are read from ``config/config.yaml``
    (section ``embedding``).

    Parameters
    ----------
    force_new:
        If ``True``, discard the cached instance and create a new one.
        Useful in tests that need to swap models.

    Returns
    -------
    FastEmbedEmbeddings
        Embedding model compatible with LangChain vector stores.
    """
    global _embeddings_instance
    if _embeddings_instance is not None and not force_new:
        return _embeddings_instance

    cfg = load_config()
    emb_cfg = cfg.get("embedding", {})
    model_name: str = emb_cfg.get("model", "BAAI/bge-small-en-v1.5")
    cache_folder: str = emb_cfg.get("cache_folder", "./embeddings_cache")

    logger.info("Initialising embeddings: model=%s", model_name)

    _embeddings_instance = FastEmbedEmbeddings(
        model_name=model_name,
        cache_folder=cache_folder,
    )

    logger.info("Embeddings ready.")
    return _embeddings_instance
