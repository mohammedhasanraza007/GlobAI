"""
retrieval/embedding_engine.py
-----------------------------
Bounded sentence embedding wrapper.
"""

from __future__ import annotations

import gc
import inspect
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, List

import numpy as np

from bootstrap.model_cache import candidate_model_roots
from core.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: Any = "cpu",
        device_kind: str = "cpu",
        cache_dir: str = "model_checkpoints",
        batch_size: int = 16,
        dimension: int = 0,
        local_files_only: bool = True,
    ):
        self.model_id = model_id
        self.device = device
        self.device_kind = device_kind
        self.cache_dir = cache_dir
        self.batch_size = max(1, min(64, int(batch_size)))
        self.local_files_only = local_files_only
        self._model: Any = None
        self._dim = max(0, int(dimension))
        self._lock = threading.RLock()

    def _sentence_transformers_device(self) -> str:
        if self.device_kind == "cuda":
            return "cuda"
        # DirectML is not a stable sentence-transformers target; keep embeddings on CPU.
        return "cpu"

    def _candidate_model_roots(self) -> list[Path]:
        return candidate_model_roots(
            Path(self.cache_dir),
            self.model_id,
            preferred_role="embeddings",
        )

    def _validate_model_source(self) -> str:
        for root in self._candidate_model_roots():
            if not root.is_dir():
                continue
            has_sentence_config = (root / "modules.json").exists()
            has_transformer_config = (root / "config.json").exists()
            if not has_sentence_config and not has_transformer_config:
                logger.warning("[EMBED] Candidate missing model config: %s", root)
                continue
            for config_name in ("modules.json", "config.json", "config_sentence_transformers.json"):
                config_path = root / config_name
                if config_path.exists():
                    try:
                        json.loads(config_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid {config_name} in {root}: {exc}") from exc
            logger.info("[EMBED] Validated local embedding folder: %s", root)
            return str(root)

        raise FileNotFoundError(
            f"No valid local embedding model found for '{self.model_id}' in cache_dir='{self.cache_dir}'."
        )

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            model_source = self._validate_model_source()
            logger.info("[EMBED] Loading embedding model %s from %s", self.model_id, model_source)
            logger.info("[EMBED] Memory before load: %s", MemoryManager.snapshot())
            from sentence_transformers import SentenceTransformer

            if self.local_files_only:
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                os.environ.setdefault("HF_HUB_OFFLINE", "1")

            kwargs: dict[str, Any] = {
                "cache_folder": self.cache_dir,
                "device": self._sentence_transformers_device(),
            }
            try:
                if "local_files_only" in inspect.signature(SentenceTransformer.__init__).parameters:
                    kwargs["local_files_only"] = self.local_files_only
            except (TypeError, ValueError):
                pass

            try:
                self._model = SentenceTransformer(model_source, **kwargs)
            except TypeError as exc:
                if "local_files_only" not in str(exc):
                    raise
                kwargs.pop("local_files_only", None)
                self._model = SentenceTransformer(model_source, **kwargs)
            sample = self._model.encode(["probe"], normalize_embeddings=True, show_progress_bar=False)
            actual_dim = int(sample.shape[1])
            if self._dim and self._dim != actual_dim:
                raise ValueError(
                    f"Configured embedding_dimension={self._dim}, but {self.model_id} produced {actual_dim}. "
                    "Update config.yaml or rebuild the vector store."
                )
            self._dim = actual_dim
            logger.info("[EMBED] Model loaded. Dimension=%d Memory=%s", self._dim, MemoryManager.snapshot())

    def unload(self) -> None:
        with self._lock:
            if self._model is not None:
                logger.info("[EMBED] Unloading embedding model %s", self.model_id)
                model = self._model
                self._model = None
                del model
            MemoryManager.hard_cleanup("embedding unload")

    def embed(self, texts: List[str]) -> np.ndarray:
        clean_texts = [str(t or "").strip() for t in texts if str(t or "").strip()]
        if not clean_texts:
            return np.zeros((0, self._dim), dtype=np.float32)

        self._ensure_loaded()
        vectors: list[np.ndarray] = []
        try:
            for start in range(0, len(clean_texts), self.batch_size):
                batch = clean_texts[start : start + self.batch_size]
                vecs = self._model.encode(
                    batch,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=self.batch_size,
                )
                vectors.append(np.asarray(vecs, dtype=np.float32))
                gc.collect()

            if not vectors:
                return np.zeros((0, self._dim), dtype=np.float32)
            return MemoryManager.normalize_embeddings(np.vstack(vectors)).astype(np.float32)
        finally:
            self.unload()

    def dimension(self) -> int:
        if self._dim:
            return self._dim
        self._ensure_loaded()
        try:
            return self._dim
        finally:
            self.unload()
