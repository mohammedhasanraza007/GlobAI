"""
core/error_classifier.py
------------------------
Structured error classification for observability and routing decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

ErrorCategory = Literal[
    "oom",
    "model_not_found",
    "timeout",
    "syntax_error",
    "token_limit",
    "device_error",
    "index_corrupt",
    "io_error",
    "generation_empty",
    "unknown",
]

ErrorSeverity = Literal["critical", "high", "medium", "low"]


@dataclass(frozen=True)
class ClassifiedError:
    category: ErrorCategory
    severity: ErrorSeverity
    original: str
    recoverable: bool
    hint: str

    def to_dict(self) -> dict:
        return {
            "error_category": self.category,
            "error_severity": self.severity,
            "error": self.original,
            "recoverable": self.recoverable,
            "hint": self.hint,
        }


_RULES: list[tuple[tuple[str, ...], ErrorCategory, ErrorSeverity, bool, str]] = [
    (
        ("out of memory", "cuda out of memory", "oom", "memory allocation"),
        "oom", "critical", True,
        "Reduce batch size, unload other modes, or free RAM before retrying.",
    ),
    (
        ("no valid local model", "model not found", "modelnotfounderror", "filenotfounderror",
         "no such file", "missing model"),
        "model_not_found", "high", True,
        "Run scripts/download_models.py manually, then retry.",
    ),
    (
        ("timed out", "timeout", "timeouterror"),
        "timeout", "high", True,
        "Increase coder_generation_timeout or use a smaller model.",
    ),
    (
        ("syntaxerror", "syntax error", "invalid syntax", "unexpected eof"),
        "syntax_error", "medium", True,
        "Code validation failed. The model will retry up to max_retries times.",
    ),
    (
        ("tokenlimiterror", "exceeds max_input_tokens", "token limit"),
        "token_limit", "medium", True,
        "Reduce max_input_tokens or shorten the prompt.",
    ),
    (
        ("directml", "cuda error", "device error", "runtime error: device"),
        "device_error", "high", False,
        "Device backend error. Check DirectML/CUDA installation.",
    ),
    (
        ("faiss", "index", "reconstruct", "vector store"),
        "index_corrupt", "high", True,
        "Delete data/vector_db and re-index documents.",
    ),
    (
        ("permissionerror", "oserror", "ioerror", "no space left"),
        "io_error", "high", False,
        "File system error. Check disk space and permissions.",
    ),
    (
        ("no response generated", "empty generation", "(no response"),
        "generation_empty", "medium", True,
        "Model produced empty output. Try rephrasing the prompt.",
    ),
]


def classify_error(exc: BaseException | str) -> ClassifiedError:
    msg = str(exc).lower()
    for keywords, category, severity, recoverable, hint in _RULES:
        if any(kw in msg for kw in keywords):
            classified = ClassifiedError(
                category=category,
                severity=severity,
                original=str(exc),
                recoverable=recoverable,
                hint=hint,
            )
            logger.warning(
                "[ERROR_CLASSIFIER] %s | severity=%s | recoverable=%s | %s",
                category.upper(),
                severity,
                recoverable,
                str(exc)[:200],
            )
            return classified

    classified = ClassifiedError(
        category="unknown",
        severity="medium",
        original=str(exc),
        recoverable=True,
        hint="Check logs for details.",
    )
    logger.warning("[ERROR_CLASSIFIER] UNKNOWN | %s", str(exc)[:200])
    return classified
