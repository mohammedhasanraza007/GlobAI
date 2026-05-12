"""
bootstrap/env_lock.py
---------------------
Python runtime guard.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

SUPPORTED = {(3, 10)}
REQUIRED_VENV_NAME = "nexarag_env"


def enforce_virtualenv() -> None:
    # Allow portable runtime (embedded distribution)
    prefix_path = Path(sys.prefix)
    if prefix_path.name.lower() == "runtime":
        print("[ENV_LOCK] Portable runtime detected. Skipping virtual environment check.")
        return

    # Standard venv check
    if sys.prefix == sys.base_prefix:
        print(
            "\n[ENV_LOCK] ABORT - virtual environment is not active.\n"
            "           Run Setup.exe, then launch GlobAI.exe.\n"
        )
        sys.exit(1)

    active_name = prefix_path.name.lower()
    if active_name != REQUIRED_VENV_NAME:
        print(
            "\n[ENV_LOCK] ABORT - wrong virtual environment is active.\n"
            f"           Required: {REQUIRED_VENV_NAME}\n"
            f"           Active  : {prefix_path.name}\n"
        )
        sys.exit(1)

    print(f"[ENV_LOCK] Virtual environment '{REQUIRED_VENV_NAME}' is active.")


def enforce_python_version() -> None:
    detected = (sys.version_info.major, sys.version_info.minor)
    full = platform.python_version()
    if detected not in SUPPORTED:
        supported = ", ".join(f"{major}.{minor}.x" for major, minor in sorted(SUPPORTED))
        print(
            f"\n[ENV_LOCK] ABORT - Python {supported} required.\n"
            f"           Detected: Python {full}\n"
            "           This DirectML lock uses torch-directml==0.2.0.dev230426, which ships cp310 wheels.\n"
        )
        sys.exit(1)
    print(f"[ENV_LOCK] Python {full} - version check passed.")


if __name__ == "__main__":
    enforce_virtualenv()
    enforce_python_version()
