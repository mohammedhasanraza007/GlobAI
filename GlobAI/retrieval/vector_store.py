"""
retrieval/vector_store.py
-------------------------
Persistent, memory-bounded FAISS vector store.

V2 additions:
- validate_consistency() for FAISS index health checks
- Duplicate chunk detection on add
- Better logging on search results
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from core.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"
LEGACY_CHUNKS_FILE = "chunks.pkl"


class VectorStoreConsistencyError(RuntimeError):
    pass


class VectorStore:
    def __init__(
        self,
        store_dir: str,
        dimension: int,
        max_vectors: int = 5000,
        dedupe_similarity: float = 0.97,
        dedupe_window: int = 128,
    ):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.dimension = int(dimension)
        self.max_vectors = int(max_vectors)
        self.dedupe_similarity = float(dedupe_similarity)
        self.dedupe_window = max(0, int(dedupe_window))
        self._idx_path = self.store_dir / INDEX_FILE
        self._chunks_path = self.store_dir / CHUNKS_FILE
        self._legacy_chunks_path = self.store_dir / LEGACY_CHUNKS_FILE
        self._lock = threading.Lock()
        self.index, self.chunks = self._load_or_create()
        self.validate_consistency(repair=True)

    def _load_chunks(self) -> list[str]:
        if self._chunks_path.exists():
            with self._chunks_path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, list):
                raise ValueError("chunks.json must contain a list.")
            return [str(item) for item in data]

        if self._legacy_chunks_path.exists():
            logger.warning(
                "[VS] Ignoring legacy chunks.pkl. Pickle cache loading is disabled for safety."
            )

        return []

    def _load_or_create(self) -> Tuple[faiss.Index, List[str]]:
        if self._idx_path.exists():
            try:
                idx = faiss.read_index(str(self._idx_path))
                chunks = self._load_chunks()
                if idx.d != self.dimension:
                    raise ValueError(f"Index dim {idx.d} != requested {self.dimension}.")
                if idx.ntotal != len(chunks):
                    aligned = min(idx.ntotal, len(chunks))
                    logger.warning("[VS] Index/chunk count mismatch; keeping first %d entries.", aligned)
                    if aligned:
                        buf = np.zeros((aligned, self.dimension), dtype=np.float32)
                        idx.reconstruct_n(0, aligned, buf)
                        idx = faiss.IndexFlatIP(self.dimension)
                        idx.add(buf)
                    else:
                        idx = faiss.IndexFlatIP(self.dimension)
                    chunks = chunks[:aligned]
                logger.info("[VS] Loaded FAISS index with %d vector(s).", idx.ntotal)
                return idx, chunks
            except Exception as exc:
                logger.warning("[VS] Could not load index (%s). Creating fresh store.", exc)

        logger.info("[VS] Creating new FAISS index (dim=%d).", self.dimension)
        return faiss.IndexFlatIP(self.dimension), []

    def validate_consistency(self, repair: bool = False) -> dict:
        """
        Validate FAISS index health and chunk alignment.
        Returns a dict with validation results.
        If repair=True, attempts to fix minor inconsistencies automatically.
        """
        issues: list[str] = []

        if self.index.d != self.dimension:
            issues.append(
                f"Index dimension mismatch: index.d={self.index.d} vs configured={self.dimension}"
            )

        if self.index.ntotal != len(self.chunks):
            msg = (f"Vector/chunk count mismatch: "
                   f"index.ntotal={self.index.ntotal} vs chunks={len(self.chunks)}")
            issues.append(msg)
            if repair:
                aligned = min(self.index.ntotal, len(self.chunks))
                logger.warning("[VS] Repair: aligning to %d entries.", aligned)
                if aligned > 0:
                    buf = np.zeros((aligned, self.dimension), dtype=np.float32)
                    self.index.reconstruct_n(0, aligned, buf)
                    self.index = faiss.IndexFlatIP(self.dimension)
                    self.index.add(buf)
                else:
                    self.index = faiss.IndexFlatIP(self.dimension)
                self.chunks = self.chunks[:aligned]

        empty_chunks = [i for i, c in enumerate(self.chunks) if not str(c).strip()]
        if empty_chunks:
            issues.append(f"{len(empty_chunks)} empty chunk(s) detected at indices: {empty_chunks[:5]}")
            if repair and empty_chunks:
                logger.warning("[VS] Repair: removing %d empty chunks.", len(empty_chunks))
                keep = [i for i in range(len(self.chunks)) if str(self.chunks[i]).strip()]
                if keep and self.index.ntotal > 0:
                    buf = np.zeros((self.index.ntotal, self.dimension), dtype=np.float32)
                    self.index.reconstruct_n(0, self.index.ntotal, buf)
                    kept_embs = buf[keep]
                    self.index = faiss.IndexFlatIP(self.dimension)
                    self.index.add(kept_embs)
                    self.chunks = [self.chunks[i] for i in keep]

        result = {
            "ok": len(issues) == 0,
            "vector_count": self.index.ntotal,
            "chunk_count": len(self.chunks),
            "issues": issues,
        }
        if issues:
            logger.warning("[VS] Consistency check: %d issue(s): %s", len(issues), issues)
        else:
            logger.debug("[VS] Consistency check passed (%d vectors).", self.index.ntotal)
        return result

    def _existing_embeddings(self) -> np.ndarray:
        if self.index.ntotal == 0:
            return np.zeros((0, self.dimension), dtype=np.float32)
        buf = np.zeros((self.index.ntotal, self.dimension), dtype=np.float32)
        self.index.reconstruct_n(0, self.index.ntotal, buf)
        return buf

    def add(self, embeddings: np.ndarray, new_chunks: List[str]) -> int:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embeddings shape (N, {self.dimension}), got {embeddings.shape}."
            )
        if embeddings.shape[0] == 0 or not new_chunks:
            return 0

        pairs = [
            (embeddings[index], str(chunk).strip())
            for index, chunk in enumerate(new_chunks[: embeddings.shape[0]])
            if str(chunk).strip()
        ]
        if not pairs:
            return 0
        new_embs = np.vstack([pair[0] for pair in pairs]).astype(np.float32)
        clean_chunks = [pair[1] for pair in pairs]

        with self._lock:
            existing = self._existing_embeddings()
            all_embs = np.vstack(
                [existing, MemoryManager.normalize_embeddings(new_embs)]
            )
            all_chunks = list(self.chunks) + clean_chunks
            all_embs, all_chunks = MemoryManager.compress_vector_store(
                all_embs,
                all_chunks,
                max_vectors=self.max_vectors,
                dedupe_similarity=self.dedupe_similarity,
                dedupe_window=self.dedupe_window,
            )

            self.index = faiss.IndexFlatIP(self.dimension)
            if len(all_embs):
                self.index.add(all_embs.astype(np.float32))
            self.chunks = all_chunks
            self._persist()
            logger.info("[VS] Store now contains %d vector(s). Added %d chunk(s).",
                        self.index.ntotal, len(clean_chunks))
            return len(clean_chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        with self._lock:
            if self.index.ntotal == 0:
                return []
            query = np.asarray(query_embedding, dtype=np.float32)
            if query.ndim == 1:
                query = query.reshape(1, -1)
            query = MemoryManager.normalize_embeddings(query)
            k = min(max(1, int(top_k)), self.index.ntotal)
            scores, indices = self.index.search(query, k)
            results: list[tuple[str, float]] = []
            for idx, score in zip(indices[0], scores[0]):
                if 0 <= idx < len(self.chunks):
                    results.append((self.chunks[int(idx)], float(score)))
            logger.debug("[VS] Search returned %d result(s). Top score=%.4f.",
                         len(results), results[0][1] if results else 0.0)
            return results

    def _persist(self) -> None:
        tmp_index = self._idx_path.with_suffix(".faiss.tmp")
        tmp_chunks = self._chunks_path.with_suffix(".json.tmp")

        faiss.write_index(self.index, str(tmp_index))
        with tmp_chunks.open("w", encoding="utf-8") as fh:
            json.dump(self.chunks, fh, ensure_ascii=False)

        os.replace(tmp_index, self._idx_path)
        os.replace(tmp_chunks, self._chunks_path)

    def count(self) -> int:
        return int(self.index.ntotal)
