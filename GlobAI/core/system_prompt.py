"""
core/system_prompt.py
---------------------
Global system prompt engine for GlobAI.

Provides a single, process-wide compatibility prompt setting.
State is held in-memory and may be persisted to config.yaml via the UI.

DO NOT import from RAG, coder, image, or indexing modules here.
"""

from __future__ import annotations

DEFAULT_SYSTEM_PROMPT: str = (
    "You are a precise, deterministic AI. "
    "Follow instructions exactly. Do not hallucinate."
)

_SYSTEM_PROMPT: str = DEFAULT_SYSTEM_PROMPT


def get_system_prompt() -> str:
    """Return the currently active system prompt."""
    return _SYSTEM_PROMPT


def set_system_prompt(new_prompt: str) -> None:
    """
    Replace the active system prompt in memory.
    Persists only until process restart unless also saved to config.
    """
    global _SYSTEM_PROMPT
    cleaned = str(new_prompt or "").strip()
    _SYSTEM_PROMPT = cleaned if cleaned else DEFAULT_SYSTEM_PROMPT

