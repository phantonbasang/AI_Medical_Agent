"""
src/models/llm.py

Centralised LLM initialisation.

All API keys are read from environment variables — never hard-coded.
Supports Groq (primary) and Google Gemini (optional fallback).

Usage
-----
    from src.models.llm import get_llm
    llm = get_llm()
    response = llm.invoke("What is diabetes?")
"""
from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

from src.utils import get_logger, load_config

logger = get_logger(__name__)

# Module-level singleton so we only initialise once per process
_llm_instance: BaseChatModel | None = None


def get_llm(force_new: bool = False) -> BaseChatModel:
    """
    Return a cached LLM instance (Groq by default).

    The model, temperature and other settings are read from
    ``config/config.yaml`` and can be overridden via env vars:

    - ``GROQ_API_KEY``   — required for Groq provider
    - ``GOOGLE_API_KEY`` — required for Gemini provider
    - ``LLM__MODEL``     — override model name at runtime

    Parameters
    ----------
    force_new:
        If ``True``, discard the cached instance and create a new one.

    Returns
    -------
    BaseChatModel
        Initialised chat model ready to call ``.invoke()`` or ``.stream()``.

    Raises
    ------
    EnvironmentError
        When the required API key is missing.
    ImportError
        When the provider package is not installed.
    """
    global _llm_instance
    if _llm_instance is not None and not force_new:
        return _llm_instance

    cfg = load_config()
    llm_cfg = cfg.get("llm", {})
    provider: str = llm_cfg.get("provider", "groq").lower()
    model: str = os.getenv("LLM__MODEL", llm_cfg.get("model", "llama-3.1-8b-instant"))
    temperature: float = float(llm_cfg.get("temperature", 0.1))
    max_tokens: int = int(llm_cfg.get("max_tokens", 2048))
    max_retries: int = int(llm_cfg.get("max_retries", 3))

    logger.info("Initialising LLM: provider=%s  model=%s", provider, model)

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Copy config/.env.example to .env and fill in your key."
            )
        try:
            from langchain_groq import ChatGroq  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "langchain-groq is not installed. Run: pip install langchain-groq"
            ) from exc

        _llm_instance = ChatGroq(
            groq_api_key=api_key,
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

    elif provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY is not set. "
                "Copy config/.env.example to .env and fill in your key."
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "langchain-google-genai is not installed. "
                "Run: pip install langchain-google-genai"
            ) from exc

        _llm_instance = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Copy config/.env.example to .env and fill in your key."
            )
        try:
            from langchain_openai import ChatOpenAI  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "langchain-openai is not installed. Run: pip install langchain-openai"
            ) from exc

        _llm_instance = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

    else:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            "Choose 'groq', 'gemini', or 'openai' in config/config.yaml."
        )

    logger.info("LLM ready: %s", model)
    return _llm_instance


def get_fallback_llm() -> BaseChatModel:
    """
    Return a fallback LLM (larger / more capable model).

    Used when the primary model fails to call tools correctly.
    Reads ``llm.fallback_model`` from config.
    """
    cfg = load_config()
    llm_cfg = cfg.get("llm", {})
    fallback_model: str = llm_cfg.get("fallback_model", "llama-3.3-70b-versatile")

    # Temporarily patch model in cfg and create a fresh instance
    original = llm_cfg.get("model")
    llm_cfg["model"] = fallback_model
    try:
        instance = get_llm(force_new=True)
    finally:
        llm_cfg["model"] = original
        # Reset singleton so next call to get_llm() uses primary
        global _llm_instance
        _llm_instance = None

    return instance
