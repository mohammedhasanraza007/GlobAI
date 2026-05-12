"""
retrieval/hybrid_retriever.py
-----------------------------
Bounded hybrid retrieval: vector cosine + BM25 keyword score.

V2 additions:
- Re-ranking layer: length normalization + score boosting for content density
- Per-chunk debug logging with scores
- Deterministic result ordering
"""

from __future__ import annotations

import gc
import logging
from typing import List, Tuple

from retrieval.bm25_retriever import BM25Retriever, _tokenize
from retrieval.embedding_engine import EmbeddingEngine
from retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

FAISS_WEIGHT = 0.6
BM25_WEIGHT = 0.4

_MIN_CHUNK_WORDS = 5
_LENGTH_BONUS_THRESHOLD = 30
_LENGTH_BONUS = 0.03
_LENGTH_BONUS_MIN_SCORE = 0.35
_MIN_BM25_SIGNAL = 0.08
_MIN_QUERY_OVERLAP = 1
_STRONG_VECTOR_SCORE = 0.42


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        embedding_engine: EmbeddingEngine,
        similarity_threshold: float = 0.20,
        top_k: int = 3,
        candidate_k: int = 10,
        max_context_chars: int = 6000,
    ):
        self.vs = vector_store
        self.bm25 = bm25_retriever
        self.embedder = embedding_engine
        self.threshold = float(similarity_threshold)
        self.top_k = max(1, min(6, int(top_k)))
        self.candidate_k = max(self.top_k, min(24, int(candidate_k)))
        self.max_context_chars = max(500, min(12_000, int(max_context_chars)))

    @staticmethod
    def _norm_cosine(score: float) -> float:
        # FAISS IndexFlatIP stores normalized embeddings, so the score is already
        # cosine similarity. Do not remap weak unrelated matches into mid scores.
        return max(0.0, min(1.0, float(score)))

    @staticmethod
    def _rerank(ranked: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """
        Re-ranking layer applied after initial hybrid scoring.
        - Penalizes very short chunks (likely headers/fragments)
        - Gives a small bonus to content-rich chunks (>= LENGTH_BONUS_THRESHOLD words)
        - Breaks ties deterministically by chunk text length (longer preferred)
        """
        reranked: list[tuple[str, float]] = []
        for chunk, score in ranked:
            words = chunk.split()
            word_count = len(words)
            if word_count < _MIN_CHUNK_WORDS:
                adjusted = score * 0.5
            elif word_count >= _LENGTH_BONUS_THRESHOLD and score >= _LENGTH_BONUS_MIN_SCORE:
                adjusted = score + _LENGTH_BONUS
            else:
                adjusted = score
            reranked.append((chunk, min(1.0, adjusted)))

        reranked.sort(key=lambda item: (item[1], len(item[0])), reverse=True)
        return reranked

    @staticmethod
    def _lexical_overlap(query_terms: set[str], chunk: str) -> int:
        if not query_terms:
            return 0
        return len(query_terms.intersection(_tokenize(chunk)))

    def retrieve(self, query: str, use_vector: bool = True) -> List[Tuple[str, float]]:
        if not query or not query.strip():
            return []

        query_terms = set(_tokenize(query))
        query_vec = None
        faiss_hits: list[tuple[str, float]] = []
        try:
            if use_vector:
                query_vec = self.embedder.embed([query])
                faiss_hits = self.vs.search(query_vec, top_k=self.candidate_k)
            bm25_hits = self.bm25.search(query, top_k=self.candidate_k)
        finally:
            if query_vec is not None:
                del query_vec
            gc.collect()

        score_map: dict[str, dict[str, float]] = {}
        for chunk, score in faiss_hits:
            slot = score_map.setdefault(chunk, {"faiss": 0.0, "bm25": 0.0})
            slot["faiss"] = max(slot["faiss"], self._norm_cosine(score))

        for chunk, score in bm25_hits:
            slot = score_map.setdefault(chunk, {"faiss": 0.0, "bm25": 0.0})
            slot["bm25"] = max(slot["bm25"], max(0.0, min(1.0, float(score))))

        ranked: list[tuple[str, float]] = []
        for chunk, scores in score_map.items():
            hybrid = (FAISS_WEIGHT * scores["faiss"]) + (BM25_WEIGHT * scores["bm25"])
            overlap = self._lexical_overlap(query_terms, chunk)
            has_keyword_signal = scores["bm25"] >= _MIN_BM25_SIGNAL or overlap >= _MIN_QUERY_OVERLAP
            strong_vector_signal = scores["faiss"] >= max(self.threshold + 0.18, _STRONG_VECTOR_SCORE)
            if not has_keyword_signal and not strong_vector_signal:
                continue
            ranked.append((chunk, hybrid))
        ranked.sort(key=lambda item: item[1], reverse=True)

        ranked = self._rerank(ranked)

        result = [(chunk, score) for chunk, score in ranked if score >= self.threshold][: self.top_k]

        mode = "hybrid" if use_vector else "keyword-only"
        if result:
            logger.info(
                "[HYBRID] %d chunk(s) via %s. Scores: %s",
                len(result),
                mode,
                ", ".join(f"{s:.3f}" for _, s in result),
            )
            for idx, (chunk, score) in enumerate(result, 1):
                preview = chunk[:120].replace("\n", " ")
                logger.debug("[HYBRID] Chunk %d | score=%.4f | words=%d | '%s...'",
                             idx, score, len(chunk.split()), preview)
        else:
            logger.info("[HYBRID] No chunks met threshold %.3f (candidates=%d).",
                        self.threshold, len(ranked))
        return result

    def format_context(self, results: List[Tuple[str, float]]) -> str:
        if not results:
            return "(No relevant context found in the local knowledge base.)"

        parts: list[str] = []
        remaining = self.max_context_chars
        for idx, (chunk, score) in enumerate(results[: self.top_k], 1):
            header = f"[Context {idx} | relevance={score:.3f}]\n"
            budget = remaining - len(header) - 8
            if budget <= 0:
                break
            text = str(chunk)
            if len(text) > budget:
                marker = "\n[truncated]"
                text = text[: max(0, budget - len(marker))].rstrip() + marker
            block = header + text
            parts.append(block)
            remaining -= len(block) + 6
        return "\n\n---\n\n".join(parts)
