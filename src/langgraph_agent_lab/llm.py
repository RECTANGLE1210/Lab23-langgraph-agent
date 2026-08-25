"""LLM factory helper.

Provides a simple interface to create LLM clients for use in nodes.
Students should use this helper so the lab works with any supported provider.

Usage in nodes:
    from .llm import get_llm
    llm = get_llm()
    response = llm.invoke("Hello")
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OPENROUTER_MODEL = "google/gemini-3.7-flash"


def _load_environment() -> None:
    """Load the repository .env without overriding explicitly-set variables."""
    load_dotenv(_REPO_ROOT / ".env", override=False)


def get_llm(model: str | None = None, temperature: float = 0.0) -> BaseChatModel:
    """Create an LLM client from environment configuration.

    Checks for API keys in this order:
    1. OPENROUTER_API_KEY → ChatOpenRouter
    2. GEMINI_API_KEY → ChatGoogleGenerativeAI
    3. OPENAI_API_KEY → ChatOpenAI
    4. ANTHROPIC_API_KEY → ChatAnthropic

    Override model with the `model` parameter or LLM_MODEL env var.
    """
    _load_environment()

    if os.getenv("OPENROUTER_API_KEY"):
        try:
            from langchain_openrouter import ChatOpenRouter
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-openrouter") from exc
        return ChatOpenRouter(
            model=model or os.getenv("LLM_MODEL") or _DEFAULT_OPENROUTER_MODEL,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=temperature,
        )

    if os.getenv("GEMINI_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-google-genai") from exc
        return ChatGoogleGenerativeAI(
            model=model or os.getenv("LLM_MODEL", "gemini-2.5-flash"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
        )

    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-openai") from exc
        return ChatOpenAI(
            model=model or os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-anthropic") from exc
        return ChatAnthropic(
            model=model or os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"),
            temperature=temperature,
        )

    raise RuntimeError(
        "No LLM API key found. Set OPENROUTER_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, "
        "or ANTHROPIC_API_KEY in .env\n"
        "See .env.example for configuration."
    )
