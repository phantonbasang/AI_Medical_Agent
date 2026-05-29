"""
setup.py — AI Medical Agent package installer.

Usage
-----
pip install -e .          # editable / development install
pip install .             # regular install
"""
from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="ai-medical-agent",
    version="0.1.0",
    description="Multi-agent RAG system for medical question answering using LangGraph and ChromaDB",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="phantonbasang",
    url="https://github.com/phantonbasang/AI_Medical_Agent",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "langchain>=0.3.0",
        "langchain-core>=0.3.0",
        "langchain-community>=0.3.0",
        "langchain-groq>=0.2.0",
        "langgraph>=0.2.0",
        "chromadb>=0.5.0",
        "fastembed>=0.3.0",
        "datasets>=3.0.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "pyyaml>=6.0.1",
        "python-dotenv>=1.0.0",
        "tenacity>=8.5.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "eval": ["ragas>=0.1.21"],
        "viz": ["matplotlib>=3.9.0", "seaborn>=0.13.0", "plotly>=5.22.0"],
        "dev": ["pytest>=8.3.0", "pytest-asyncio>=0.24.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    keywords="medical ai agent langgraph rag chromadb llm groq pubmedqa",
)
