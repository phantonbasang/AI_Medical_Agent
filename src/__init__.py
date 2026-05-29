"""
src/__init__.py

AI Medical Agent — top-level package.

Exposes the high-level public API so callers can do:

    from src import load_config, get_logger
    from src.agents.graph import build_medical_graph
"""
from src.utils import load_config, get_logger  # noqa: F401

__version__ = "0.1.0"
__all__ = ["load_config", "get_logger", "__version__"]
