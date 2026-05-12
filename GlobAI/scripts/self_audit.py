"""
scripts/self_audit.py
---------------------
Isolation audit for the separated RAG, coder, and image subsystems.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import DEFAULT_CONFIG
from coder.coder_system import CoderSystem
from core.memory_manager import MemoryManager
from image.image_system import ImageSystem
from rag.rag_system import RagSystem


def _audit_static() -> list[str]:
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        cfg = dict(DEFAULT_CONFIG)
        cfg["offline_mode"] = True
        cache_dir = str(Path(tmp) / "models")
        vector_dir = str(Path(tmp) / "vectors")

        rag = RagSystem(cfg, cache_dir, vector_dir, "cpu", "cpu", "cpu")
        coder = CoderSystem(cfg, cache_dir, "cpu")
        image = ImageSystem(cfg, cache_dir)

        before = MemoryManager.snapshot()
        for _ in range(10):
            rag.unload()
            coder.unload()
            image.unload()
        after = MemoryManager.snapshot()

        if rag.model_loader is coder.loader:
            failures.append("RAG and coder share a loader object.")
        if image.manager is rag.model_loader:
            failures.append("Image manager shares a RAG loader object.")
        if hasattr(coder, "vector_store") or hasattr(coder, "embedding_engine"):
            failures.append("Coder exposes RAG storage/embedding state.")
        if hasattr(image, "vector_store") or hasattr(image, "embedding_engine"):
            failures.append("Image exposes RAG storage/embedding state.")
        if rag.audit_state()["embedding_loaded"]:
            failures.append("RAG embedding model stayed loaded after audit unload.")
        if MemoryManager.memory_increased(before, after, ram_tolerance_mb=512.0):
            failures.append("Memory increased across repeated unload cycles.")

    return failures


def _audit_live() -> list[str]:
    from app import build_app

    failures: list[str] = []
    app = build_app(skip_preflight=True)

    indexed = app.rag_system.index_texts(["Codex isolation audit document. The answer is isolation-ok."])
    if indexed <= 0:
        failures.append("RAG indexing failed.")
    rag_result = app.route("What is the audit answer?")
    if rag_result.get("error"):
        failures.append(f"RAG query failed: {rag_result['error']}")
    if app.coder_system.loader.is_loaded() or app.image_system.manager.image_model_loaded:
        failures.append("Coder or image loaded during RAG audit.")

    coder = app.enable_coder_mode()
    if not coder.get("ok"):
        failures.append(f"Coder enable failed: {coder.get('error')}")
    app.disable_coder_mode()
    if app.rag_system.embedding_engine._model is not None:
        failures.append("Coder audit touched RAG embeddings.")

    image = app.load_sd_model()
    if not image.get("ok"):
        failures.append(f"Image load failed: {image.get('error')}")
    app.unload_sd_model()
    if app.coder_system.loader.is_loaded():
        failures.append("Image audit touched coder loader.")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="run heavyweight model-backed audit")
    args = parser.parse_args()

    failures = _audit_live() if args.live else _audit_static()
    if failures:
        print("[SELF_AUDIT] FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("[SELF_AUDIT] PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
