"""
core/memory_manager.py
----------------------
Small, explicit memory utilities for bounded local inference.

V2 additions:
- stabilize_after_unload(): post-unload RAM stabilization delay + aggressive GC
- checkpoint(): explicit GC checkpoint with logging
- DirectML cache clearing in hard_cleanup and clear_cycle
- RAM ceiling enforcement (80%)
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import psutil

logger = logging.getLogger(__name__)

RAM_CEILING_PCT: int = 80
_STABILIZE_DELAY_SECS: float = 1.0


class TokenLimitError(ValueError):
    """Raised when a request would exceed the configured token budget."""


class DuplicateModelLoadError(RuntimeError):
    """Raised when the same loaded model is requested for loading again."""


class ModelCleanupError(RuntimeError):
    """Raised when hard cleanup cannot prove the runtime is safe to continue."""


class RamCeilingError(RuntimeError):
    """Raised when RAM usage exceeds the configured ceiling."""


@dataclass(frozen=True)
class MemorySnapshot:
    ram_used_mb: float
    ram_total_mb: float
    gpu_used_mb: Optional[float]
    gpu_total_mb: Optional[float]

    @property
    def ram_used_pct(self) -> float:
        if self.ram_total_mb <= 0:
            return 0.0
        return (self.ram_used_mb / self.ram_total_mb) * 100.0

    def __str__(self) -> str:
        gpu = (
            f"GPU {self.gpu_used_mb:.0f}/{self.gpu_total_mb:.0f} MB"
            if self.gpu_used_mb is not None and self.gpu_total_mb is not None
            else "GPU n/a"
        )
        return f"RAM {self.ram_used_mb:.0f}/{self.ram_total_mb:.0f} MB ({self.ram_used_pct:.1f}%) | {gpu}"


class MemoryManager:
    @staticmethod
    def _compact_platform_heap() -> None:
        if os.name != "nt":
            return
        try:
            import ctypes
            ctypes.CDLL("msvcrt")._heapmin()
        except Exception:
            logger.debug("[MEM] Windows heap compaction skipped.", exc_info=True)

    @staticmethod
    def _clear_directml_cache() -> None:
        try:
            import torch_directml  # type: ignore
            if hasattr(torch_directml, "empty_cache"):
                torch_directml.empty_cache()
        except Exception:
            pass

    @staticmethod
    def snapshot() -> MemorySnapshot:
        vm = psutil.virtual_memory()
        ram_used = (vm.total - vm.available) / (1024 * 1024)
        ram_total = vm.total / (1024 * 1024)

        gpu_used = gpu_total = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu_used = torch.cuda.memory_allocated() / (1024 * 1024)
                props = torch.cuda.get_device_properties(0)
                gpu_total = props.total_memory / (1024 * 1024)
        except Exception:
            pass

        return MemorySnapshot(ram_used, ram_total, gpu_used, gpu_total)

    @staticmethod
    def over_ram_limit(limit_pct: int) -> bool:
        return MemoryManager.snapshot().ram_used_pct >= float(limit_pct)

    @staticmethod
    def enforce_ram_ceiling(label: str = "") -> None:
        snap = MemoryManager.snapshot()
        if snap.ram_used_pct >= RAM_CEILING_PCT:
            suffix = f" ({label})" if label else ""
            logger.error(
                "[MEM] RAM ceiling exceeded%s: %.1f%% >= %d%%",
                suffix, snap.ram_used_pct, RAM_CEILING_PCT,
            )
            raise RamCeilingError(
                f"RAM usage {snap.ram_used_pct:.1f}% exceeds ceiling {RAM_CEILING_PCT}%{suffix}."
            )

    @staticmethod
    def checkpoint(label: str = "") -> MemorySnapshot:
        """Forced GC checkpoint — call at explicit safe points in the pipeline."""
        gc.collect()
        MemoryManager._clear_directml_cache()
        snap = MemoryManager.snapshot()
        suffix = f" [{label}]" if label else ""
        logger.info("[MEM] Checkpoint%s: %s", suffix, snap)
        if snap.ram_used_pct >= RAM_CEILING_PCT:
            logger.warning("[MEM] RAM ceiling exceeded at checkpoint%s.", suffix)
        return snap

    @staticmethod
    def _clear_all_torch_caches() -> None:
        """Clear torch memory caches on ALL backends (CUDA, DirectML, CPU)."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
        except Exception:
            pass
        MemoryManager._clear_directml_cache()

    @staticmethod
    def stabilize_after_unload(label: str = "") -> MemorySnapshot:
        """
        Post-unload RAM stabilization: multiple aggressive GC passes + brief delay.
        Call this after any model unload to allow the OS to reclaim memory.
        """
        before_snap = MemoryManager.snapshot()
        suffix = f" ({label})" if label else ""
        
        # Phase 1: aggressive GC + cache clearing
        for _ in range(5):
            gc.collect()
        MemoryManager._compact_platform_heap()
        MemoryManager._clear_all_torch_caches()

        # Phase 2: wait for OS to reclaim pages
        time.sleep(_STABILIZE_DELAY_SECS)

        # Phase 3: second pass to catch freed cycles
        for _ in range(3):
            gc.collect()
        MemoryManager._compact_platform_heap()

        snap = MemoryManager.snapshot()
        logger.info("[MEM] Stabilized after unload%s. %s", suffix, snap)
        return snap

    @staticmethod
    def emergency_reclaim(label: str = "") -> MemorySnapshot:
        """
        Maximum-effort memory reclamation for failed model loads.
        Clears diffusers/transformers internal module caches, forces multiple
        GC passes with delays, and compacts the heap repeatedly.
        """
        suffix = f" ({label})" if label else ""
        logger.warning("[MEM] Emergency reclaim started%s.", suffix)

        # Clear any lingering torch modules from sys.modules cache
        import sys as _sys
        modules_to_purge = [
            k for k in list(_sys.modules.keys())
            if k.startswith(("diffusers.", "transformers."))
            and "pipeline" in k.lower()
        ]
        for mod_name in modules_to_purge:
            try:
                del _sys.modules[mod_name]
            except KeyError:
                pass

        # Multiple rounds of GC + heap compaction with delays
        for i in range(3):
            for _ in range(5):
                gc.collect()
            MemoryManager._clear_all_torch_caches()
            MemoryManager._compact_platform_heap()
            time.sleep(0.5)

        snap = MemoryManager.snapshot()
        logger.info("[MEM] Emergency reclaim complete%s. %s", suffix, snap)
        return snap

    @staticmethod
    def clear_cycle() -> MemorySnapshot:
        gc.collect()
        MemoryManager._compact_platform_heap()
        MemoryManager._clear_all_torch_caches()
        snap = MemoryManager.snapshot()
        logger.info("[MEM] Post-cycle cleanup complete. %s", snap)
        if snap.ram_used_pct >= RAM_CEILING_PCT:
            logger.error(
                "[MEM] RAM ceiling exceeded after clear_cycle: %.1f%% >= %d%%",
                snap.ram_used_pct, RAM_CEILING_PCT,
            )
        return snap

    @staticmethod
    def hard_cleanup(label: str = "") -> MemorySnapshot:
        for _ in range(3):
            gc.collect()
        MemoryManager._compact_platform_heap()
        MemoryManager._clear_all_torch_caches()
        snap = MemoryManager.snapshot()
        suffix = f" ({label})" if label else ""
        logger.info("[MEM] Hard cleanup complete%s. %s", suffix, snap)
        if snap.ram_used_pct >= RAM_CEILING_PCT:
            logger.error(
                "[MEM] RAM ceiling exceeded after hard_cleanup%s: %.1f%% >= %d%%",
                suffix, snap.ram_used_pct, RAM_CEILING_PCT,
            )
        return snap

    @staticmethod
    def memory_increased(
        before: MemorySnapshot,
        after: MemorySnapshot,
        ram_tolerance_mb: float = 256.0,
        gpu_tolerance_mb: float = 64.0,
    ) -> bool:
        if after.ram_used_mb > before.ram_used_mb + ram_tolerance_mb:
            return True
        if before.gpu_used_mb is not None and after.gpu_used_mb is not None:
            return after.gpu_used_mb > before.gpu_used_mb + gpu_tolerance_mb
        return False

    @staticmethod
    def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        if embeddings.size == 0:
            return embeddings.astype(np.float32)
        embs = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embs / norms

    @staticmethod
    def compress_vector_store(
        embeddings: np.ndarray,
        chunks: list[str],
        max_vectors: int,
        dedupe_similarity: float,
        dedupe_window: int = 128,
    ) -> tuple[np.ndarray, list[str]]:
        if embeddings.size == 0 or not chunks:
            return np.zeros((0, embeddings.shape[1] if embeddings.ndim == 2 else 0), dtype=np.float32), []

        embs = MemoryManager.normalize_embeddings(embeddings)
        aligned_len = min(len(embs), len(chunks))
        embs = embs[:aligned_len]
        chunks = list(chunks[:aligned_len])

        max_candidates = max_vectors * 2
        if len(embs) > max_candidates:
            drop = len(embs) - max_candidates
            embs = embs[drop:]
            chunks = chunks[drop:]
            logger.info("[MEM] Dropped %d oldest vectors before dedupe budget pass.", drop)

        seen_text: set[str] = set()
        kept_embs: list[np.ndarray] = []
        kept_chunks: list[str] = []
        window = max(0, int(dedupe_window))

        for emb, chunk in zip(embs, chunks):
            text_key = " ".join(chunk.lower().split())
            if text_key in seen_text:
                continue
            if window and kept_embs:
                recent = np.vstack(kept_embs[-window:])
                if float(np.max(recent @ emb)) >= dedupe_similarity:
                    continue
            seen_text.add(text_key)
            kept_embs.append(emb)
            kept_chunks.append(chunk)

        if kept_embs:
            filtered_embs = np.vstack(kept_embs).astype(np.float32)
        else:
            filtered_embs = np.zeros((0, embs.shape[1]), dtype=np.float32)
        filtered_chunks = kept_chunks

        removed = len(embs) - len(filtered_chunks)
        if removed:
            logger.info("[MEM] Removed %d near-duplicate vectors.", removed)

        if len(filtered_embs) > max_vectors:
            excess = len(filtered_embs) - max_vectors
            filtered_embs = filtered_embs[excess:]
            filtered_chunks = filtered_chunks[excess:]
            logger.info("[MEM] Pruned %d oldest vectors to enforce max=%d.", excess, max_vectors)

        return filtered_embs.astype(np.float32), filtered_chunks


@dataclass
class SemanticMemoryEntry:
    content: str
    embedding: np.ndarray
    importance_score: float
    last_used: float = field(default_factory=time.time)


class Level3Memory:
    """
    Strict three-tier memory with no background work and no full-history injection.
    V2: aggressive compression pass on overflow.
    """

    def __init__(
        self,
        max_token_limit: int = 4096,
        semantic_dedupe_similarity: float = 0.97,
        tokenizer: Callable[[str], list[int]] | None = None,
    ):
        self.max_token_limit = max(1, int(max_token_limit))
        self.semantic_dedupe_similarity = float(semantic_dedupe_similarity)
        self.tokenizer = tokenizer
        self.short_term_memory: list[dict[str, str]] = []
        self.long_term_memory: list[str] = []
        self.semantic_memory: list[SemanticMemoryEntry] = []
        self._lock = threading.RLock()

    def _token_count(self, text: str) -> int:
        text = str(text or "")
        if not text:
            return 0
        if self.tokenizer is not None:
            return len(self.tokenizer(text))
        return max(1, len(text.split()))

    def _total_tokens(self) -> int:
        short = sum(self._token_count(item["content"]) for item in self.short_term_memory)
        long = sum(self._token_count(item) for item in self.long_term_memory)
        semantic = sum(self._token_count(item.content) for item in self.semantic_memory)
        return short + long + semantic

    def _summarize_oldest(self) -> bool:
        if not self.short_term_memory:
            return False
        oldest = self.short_term_memory.pop(0)
        words = oldest["content"].split()
        if not words:
            return True
        budget = max(1, min(96, self.max_token_limit // 8))
        summary = " ".join(words[:budget])
        self.long_term_memory.append(f"{oldest['role']}: {summary}")
        logger.info("[L3MEM] Compressed one short-term memory into long-term memory.")
        return True

    def compress(self) -> None:
        """Aggressive compression: trim long-term memory to half capacity."""
        with self._lock:
            while self._total_tokens() > self.max_token_limit // 2:
                if self.long_term_memory:
                    self.long_term_memory.pop(0)
                elif self.short_term_memory:
                    self.short_term_memory.pop(0)
                else:
                    break
            gc.collect()
            logger.info("[L3MEM] Aggressive compression complete. Tokens: %d", self._total_tokens())

    def add_message(self, role: str, content: str) -> None:
        role = str(role or "").strip()
        content = str(content or "").strip()
        if not role or not content:
            return
        if self._token_count(content) > self.max_token_limit:
            raise TokenLimitError("Memory entry exceeds max_token_limit.")

        with self._lock:
            entry = {"role": role, "content": content}
            self.short_term_memory.append(entry)
            while self._total_tokens() > self.max_token_limit:
                if not self._summarize_oldest():
                    if entry in self.short_term_memory:
                        self.short_term_memory.remove(entry)
                    raise TokenLimitError("Memory exceeds max_token_limit.")
            if self._total_tokens() > self.max_token_limit:
                if entry in self.short_term_memory:
                    self.short_term_memory.remove(entry)
                raise TokenLimitError("Memory exceeds max_token_limit.")

    def add_semantic(
        self,
        content: str,
        embedding: np.ndarray,
        importance_score: float = 0.5,
    ) -> bool:
        content = str(content or "").strip()
        if not content:
            return False
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError("Semantic memory embedding must be a 1-D vector.")
        vector = MemoryManager.normalize_embeddings(vector.reshape(1, -1))[0]

        with self._lock:
            for index, entry in enumerate(self.semantic_memory):
                if entry.embedding.shape == vector.shape:
                    similarity = float(entry.embedding @ vector)
                    if similarity > self.semantic_dedupe_similarity:
                        if importance_score > entry.importance_score:
                            self.semantic_memory[index] = SemanticMemoryEntry(
                                content=content,
                                embedding=vector,
                                importance_score=float(importance_score),
                                last_used=time.time(),
                            )
                        return False
            self.semantic_memory.append(
                SemanticMemoryEntry(
                    content=content,
                    embedding=vector,
                    importance_score=float(importance_score),
                    last_used=time.time(),
                )
            )
            while self._total_tokens() > self.max_token_limit:
                if not self._summarize_oldest():
                    removed = self.semantic_memory.pop()
                    logger.warning("[L3MEM] Rejected semantic memory over token budget: %s",
                                   removed.content)
                    raise TokenLimitError("Memory exceeds max_token_limit.")
            return True

    def retrieve_semantic(self, query_embedding: np.ndarray, top_k: int = 3) -> list[SemanticMemoryEntry]:
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim != 1:
            raise ValueError("Query embedding must be a 1-D vector.")
        query = MemoryManager.normalize_embeddings(query.reshape(1, -1))[0]
        top_k = max(0, int(top_k))
        if top_k == 0:
            return []

        with self._lock:
            scored: list[tuple[float, SemanticMemoryEntry]] = []
            for entry in self.semantic_memory:
                if entry.embedding.shape != query.shape:
                    continue
                score = float(entry.embedding @ query)
                scored.append((score * max(0.0, entry.importance_score), entry))
            scored.sort(key=lambda item: item[0], reverse=True)
            selected = [entry for _, entry in scored[:top_k]]
            now = time.time()
            selected_ids = {id(entry) for entry in selected}
            for entry in self.semantic_memory:
                if id(entry) in selected_ids:
                    entry.last_used = now
            logger.info("[L3MEM] Retrieved %d semantic memor(y/ies).", len(selected))
            return selected

    def semantic_context(self, query_embedding: np.ndarray, top_k: int = 3) -> str:
        entries = self.retrieve_semantic(query_embedding, top_k=top_k)
        if not entries:
            return ""
        return "\n".join(f"- {entry.content}" for entry in entries)
