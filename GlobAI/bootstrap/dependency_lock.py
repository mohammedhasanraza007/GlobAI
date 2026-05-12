"""
bootstrap/dependency_lock.py
----------------------------
Runtime dependency verification.
"""

from __future__ import annotations

import importlib.metadata as meta
import importlib
import logging
import sys
from typing import Dict, Optional

logger = logging.getLogger(__name__)

REQUIRED_PACKAGES: Dict[str, Optional[str]] = {
    "torch": "2.3.1",
    "torch-directml": "0.2.4.dev240815",
    "transformers": "4.40.2",
    "accelerate": "0.29.3",
    "sentencepiece": "0.2.0",
    "safetensors": "0.4.2",
    "huggingface_hub": "0.22.2",
    "diffusers": "0.27.2",
    "Pillow": "10.3.0",
    "sentence-transformers": "2.6.1",
    "numpy": "1.26.4",
    "pandas": "2.0.3",
    "scikit-learn": "1.4.2",
    "faiss-cpu": "1.7.4",
    "pypdf": "4.1.0",
    "pymupdf": "1.24.2",
    "python-docx": "1.1.0",
    "python-pptx": "0.6.23",
    "psutil": "5.9.8",
    "pyyaml": "6.0.1",
    "rank_bm25": "0.2.2",
    "PyQt6": "6.7.0",
}

IMPORT_NAME_MAP = {
    "faiss-cpu": "faiss",
    "pymupdf": "fitz",
    "python-docx": "docx",
    "python-pptx": "pptx",
    "pyyaml": "yaml",
    "Pillow": "PIL",
    "scikit-learn": "sklearn",
    "torch-directml": "torch_directml",
}


def _installed_version(pkg: str) -> str | None:
    try:
        return meta.version(pkg)
    except meta.PackageNotFoundError:
        alt = IMPORT_NAME_MAP.get(pkg)
        if not alt:
            return None
        try:
            return meta.version(alt)
        except meta.PackageNotFoundError:
            return None


def check_dependencies() -> None:
    missing: list[str] = []
    wrong_version: list[str] = []
    import_failed: list[str] = []

    for pkg, required_version in REQUIRED_PACKAGES.items():
        installed = _installed_version(pkg)
        if installed is None:
            missing.append(pkg)
            continue
        if required_version:
            if installed != required_version:
                wrong_version.append(f"{pkg}: required {required_version}, installed {installed}")
        import_name = IMPORT_NAME_MAP.get(pkg, pkg.replace("-", "_"))
        try:
            importlib.import_module(import_name)
        except Exception as exc:
            import_failed.append(f"{pkg} ({import_name}): {exc}")

    if missing or wrong_version or import_failed:
        print("\n[DEP_LOCK] ABORT - dependency issues detected:")
        if missing:
            print("  Missing packages:")
            for pkg in missing:
                print(f"    - {pkg}")
        if wrong_version:
            print("  Version mismatch:")
            for item in wrong_version:
                print(f"    - {item}")
        if import_failed:
            print("  Import failures:")
            for item in import_failed:
                print(f"    - {item}")
        print("\n  Run Setup.exe or pip install -r requirements.txt inside nexarag_env.\n")
        sys.exit(1)

    print(f"[DEP_LOCK] All {len(REQUIRED_PACKAGES)} dependencies verified.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check_dependencies()
