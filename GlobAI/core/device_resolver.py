"""
core/device_resolver.py
-----------------------
Central device resolution for CUDA, DirectML, and CPU.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _probe_directml() -> Tuple[Optional[Any], bool]:
    try:
        import torch_directml  # type: ignore

        if hasattr(torch_directml, "is_available") and not torch_directml.is_available():
            return None, False
        return torch_directml.device(), True
    except Exception as exc:
        logger.debug("DirectML probe failed: %s", exc)
        return None, False


def _directml_name() -> str:
    try:
        import torch_directml  # type: ignore

        return torch_directml.device_name(0) or "DirectML"
    except Exception:
        return "DirectML"


def resolve_device(preference: str = "auto") -> Tuple[Any, str]:
    pref = (preference or "auto").strip().lower()

    if pref == "cpu":
        logger.info("Device forced to CPU.")
        return torch.device("cpu"), "cpu"

    if pref in {"cuda", "gpu", "auto"} and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info("Using CUDA device: %s", name)
        return torch.device("cuda"), "cuda"

    if pref == "cuda":
        logger.warning("CUDA was requested but is unavailable. Falling back to CPU.")
        return torch.device("cpu"), "cpu"

    if pref in {"directml", "dml", "auto", "gpu"}:
        dev, ok = _probe_directml()
        if ok:
            logger.info("Using DirectML device: %s", _directml_name())
            return dev, "directml"
        if pref in {"directml", "dml"}:
            logger.warning("DirectML was requested but is unavailable. Falling back to CPU.")

    logger.info("Falling back to CPU.")
    return torch.device("cpu"), "cpu"


def log_device_banner(resolved_device: Any, resolved_kind: str, label: str = "Runtime") -> None:
    import sys

    lines = [
        "=" * 56,
        f"  GlobAI - {label} Device Diagnostics",
        "=" * 56,
        f"  Python        : {sys.version.split()[0]}",
        f"  Torch         : {torch.__version__}",
        f"  CUDA available: {torch.cuda.is_available()}",
        f"  Resolved kind : {resolved_kind}",
        f"  Device handle : {resolved_device}",
    ]
    if torch.cuda.is_available():
        lines.append(f"  GPU name      : {torch.cuda.get_device_name(0)}")
    try:
        import torch_directml as tdml  # type: ignore

        lines.append(f"  DirectML ver  : {getattr(tdml, '__version__', 'installed')}")
    except Exception:
        lines.append("  DirectML      : not installed")
    lines.append("=" * 56)
    for line in lines:
        print(line)
        logger.info(line)
