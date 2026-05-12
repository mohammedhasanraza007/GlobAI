"""
retrieval/text_chunker.py
-------------------------
Validated word chunking with bounded overlap.
Pandas is used ONLY during the indexing (chunk creation) phase and is
released immediately after cleaning.
"""

from __future__ import annotations

import gc
import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# ── Pandas runtime status flag ────────────────────────────────────────────────
# Set to True the first time _clean_texts_with_pandas actually executes.
# Used by the UI to display a real-time "Pandas: ENABLED / DISABLED" indicator.
_PANDAS_ACTIVE: bool = False


def get_pandas_status() -> bool:
    """Return True only if pandas cleaning has been executed in this session."""
    return _PANDAS_ACTIVE


def _clean_texts_with_pandas(texts: List[str]) -> List[str]:
    global _PANDAS_ACTIVE
    _PANDAS_ACTIVE = True
    import pandas as pd

    df = pd.DataFrame({"text": [str(t) for t in texts]})

    df["text"] = df["text"].str.strip()
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset="text")

    df["text"] = df["text"].str.replace(r"[ \t]+", " ", regex=True)
    df["text"] = df["text"].apply(lambda t: " ".join(t.splitlines()))
    df["text"] = df["text"].str.replace(r"[^\S\n]+", " ", regex=True).str.strip()

    total = len(df)
    if total > 1:
        line_counts: dict[str, int] = {}
        for entry in df["text"]:
            for line in entry.split(". "):
                key = line.strip().lower()
                if key:
                    line_counts[key] = line_counts.get(key, 0) + 1
        threshold = max(2, int(total * 0.3))
        repeated = {k for k, v in line_counts.items() if v >= threshold}

        def _strip_repeated(t: str) -> str:
            sentences = t.split(". ")
            cleaned = [s for s in sentences if s.strip().lower() not in repeated]
            return ". ".join(cleaned).strip()

        df["text"] = df["text"].apply(_strip_repeated)
        df["text"] = df["text"].str.strip()
        df = df[df["text"] != ""]

    cleaned = df["text"].tolist()

    del df
    gc.collect()

    return cleaned


class TextChunker:
    def __init__(self, chunk_size: int = 220, chunk_overlap: int = 40):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self._step = self.chunk_size - self.chunk_overlap

    def chunk(self, text: str) -> List[str]:
        words = str(text or "").split()
        if not words:
            return []

        chunks: list[str] = []
        for start in range(0, len(words), self._step):
            end = min(start + self.chunk_size, len(words))
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(words):
                break

        logger.debug("[CHUNKER] %d words -> %d chunks.", len(words), len(chunks))
        return chunks

    def chunk_documents(self, texts: List[str]) -> List[str]:
        cleaned = _clean_texts_with_pandas(texts)

        all_chunks: list[str] = []
        for text in cleaned:
            all_chunks.extend(self.chunk(text))
        logger.info("[CHUNKER] %d document part(s) -> %d chunk(s).", len(texts), len(all_chunks))
        return all_chunks
