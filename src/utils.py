"""
src/utils.py

Shared helpers used across every module:
  - load_config()   : Parse config/config.yaml + env-var overrides
  - get_logger()    : Consistent rotating-file + stream logger
  - safe_extract_content() : Normalise multi-type LLM responses
"""
from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load .env from the project root (if present)
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
_cached_config: dict[str, Any] | None = None


def load_config(path: str | Path = _CONFIG_PATH) -> dict[str, Any]:
    """
    Load and cache config/config.yaml.

    Environment variables can override any leaf value by matching the key
    pattern  SECTION__KEY  (double underscore), e.g.
        LLM__MODEL=llama-3.3-70b-versatile overrides config.llm.model

    Parameters
    ----------
    path:
        Filesystem path to the YAML config file.  Defaults to
        ``config/config.yaml`` relative to the project root.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    global _cached_config
    if _cached_config is not None:
        return _cached_config

    with open(path, "r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    # Apply env-var overrides (SECTION__KEY format)
    for env_key, env_val in os.environ.items():
        parts = env_key.lower().split("__")
        if len(parts) == 2:
            section, key = parts
            if section in cfg and isinstance(cfg[section], dict) and key in cfg[section]:
                # Attempt numeric cast
                try:
                    env_val = int(env_val)  # type: ignore[assignment]
                except ValueError:
                    try:
                        env_val = float(env_val)  # type: ignore[assignment]
                    except ValueError:
                        pass
                cfg[section][key] = env_val

    _cached_config = cfg
    return cfg


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str = "medical_agent") -> logging.Logger:
    """
    Return a named logger with rotating-file + stream handlers.

    The log file path and level are read from ``config.yaml``.  Repeated
    calls with the same *name* return the same logger instance.

    Parameters
    ----------
    name:
        Logger name.  Use module ``__name__`` for automatic naming.

    Returns
    -------
    logging.Logger
    """
    if name in _loggers:
        return _loggers[name]

    cfg = load_config()
    log_cfg = cfg.get("logging", {})
    level_str: str = os.getenv("LOG_LEVEL", log_cfg.get("level", "INFO"))
    level = getattr(logging, level_str.upper(), logging.INFO)
    log_file: str = log_cfg.get("file", "./medical_agent.log")
    fmt: str = log_cfg.get(
        "format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    max_bytes: int = int(log_cfg.get("max_bytes", 10 * 1024 * 1024))
    backup_count: int = int(log_cfg.get("backup_count", 3))

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(fmt)

        # Stream handler (stdout)
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        # Rotating file handler
        try:
            fh = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except OSError:
            # Running in read-only FS (e.g. CI) — stream only
            pass

    logger.propagate = False
    _loggers[name] = logger
    return logger


# ---------------------------------------------------------------------------
# LLM response helpers
# ---------------------------------------------------------------------------

def safe_extract_content(response: Any) -> str:
    """
    Normalise an LLM response to a plain string.

    Different providers return different types:
    - Groq / OpenAI   → ``AIMessage`` with ``.content`` as ``str``
    - Gemini          → ``.content`` as ``list[dict]``

    Parameters
    ----------
    response:
        Raw value returned by ``llm.invoke(...)`` or ``.content``.

    Returns
    -------
    str
        Plain-text content extracted from the response.
    """
    # Already a plain string
    if isinstance(response, str):
        return response.strip()

    # AIMessage or similar object with .content
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", str(item)))
            else:
                parts.append(str(item))
        return " ".join(parts).strip()

    return str(content).strip()
