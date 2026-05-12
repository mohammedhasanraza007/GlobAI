"""
bootstrap/preflight_check.py
----------------------------
Master preflight check.
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from bootstrap.dependency_lock import check_dependencies
from bootstrap.env_lock import enforce_python_version, enforce_virtualenv
from bootstrap.hardware_guard import check_hardware

logger = logging.getLogger(__name__)


def run_preflight() -> dict:
    print("\n" + "=" * 58)
    print("  GlobAI - PREFLIGHT CHECK")
    print("=" * 58)
    enforce_virtualenv()
    enforce_python_version()
    hw = check_hardware()
    check_dependencies()
    print("=" * 58)
    print("  ALL PREFLIGHT CHECKS PASSED")
    print("=" * 58 + "\n")
    return hw


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_preflight()
