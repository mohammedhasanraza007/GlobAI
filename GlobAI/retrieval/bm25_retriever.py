"""
retrieval/bm25_retriever.py
---------------------------
BM25 keyword retrieval rebuilt from bounded chunk text.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

BM25_CHUNKS_FILE = "bm25_chunks.json"
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could",
    "do", "does", "explain", "for", "from", "give", "how", "in", "is",
    "it", "me", "of", "on", "or", "please", "show", "tell", "that",
    "the", "this", "to", "what", "when", "where", "which", "who", "why",
    "with", "would", "you",
}


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_RE.findall(str(text).lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


class BM25Retriever:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path = self.cache_dir / BM25_CHUNKS_FILE
        self._bm25 = None
        self._chunks: List[str] = []
        self._lock = threading.Lock()
        self._load_cache()

    def _load_cache(self) -> None:
        if not self._cache_path.exists():
            return
        try:
            with self._cache_path.open(encoding="utf-8") as fh:
                chunks = json.load(fh)
            if isinstance(chunks, list) and chunks:
                self.sync([str(c) for c in chunks])
                logger.info("[BM25] Loaded %d cached chunk(s).", len(chunks))
        except Exception as exc:
            logger.warning("[BM25] Cache load failed: %s", exc)
            self._bm25 = None
            self._chunks = []

    def _save_cache(self) -> None:
        tmp = self._cache_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self._chunks, fh, ensure_ascii=False)
        os.replace(tmp, self._cache_path)

    def sync(self, chunks: List[str]) -> None:
        clean_chunks = [str(chunk).strip() for chunk in chunks if str(chunk).strip()]
        with self._lock:
            if clean_chunks == self._chunks:
                logger.info("[BM25] Index already current: %d chunk(s).", len(clean_chunks))
                return
            if not clean_chunks:
                self._bm25 = None
                self._chunks = []
                self._save_cache()
                logger.info("[BM25] Cleared empty index.")
                return
            from rank_bm25 import BM25Okapi

            tokenized = [_tokenize(chunk) for chunk in clean_chunks]
            self._bm25 = BM25Okapi(tokenized)
            self._chunks = clean_chunks
            self._save_cache()
            logger.info("[BM25] Index rebuilt: %d chunk(s).", len(clean_chunks))

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        with self._lock:
            if self._bm25 is None or not self._chunks:
                return []
            tokens = _tokenize(query)
            if not tokens:
                return []
            raw_scores = np.asarray(self._bm25.get_scores(tokens), dtype=np.float32)
            max_score = float(raw_scores.max()) if raw_scores.size else 0.0
            if max_score <= 0:
                return []
            norm_scores = raw_scores / max_score
            limit = min(max(1, int(top_k)), len(self._chunks))
            top_indices = norm_scores.argsort()[::-1][:limit]
            return [
                (self._chunks[int(i)], float(norm_scores[int(i)]))
                for i in top_indices
                if norm_scores[int(i)] > 0
            ]
