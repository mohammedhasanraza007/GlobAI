"""
core/query_engine.py
--------------------
Thread-safe LOAD -> PROCESS -> RESPOND -> CLEAN -> RESET query pipeline.

V2 additions:
- Query rewriting step for better retrieval coverage
- Per-chunk debug logging (retrieved chunks + similarity scores)
- Error classification on failures
- Fail-safe returns (never crashes main loop)
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Dict

from core.error_classifier import classify_error
from core.memory_manager import MemoryManager
from core.model_loader import ModelLoader
from core.state_machine import QueryStateMachine, State
from retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 4000

_FILLER = re.compile(
    r"^\s*(please|can you|could you|would you|tell me|show me|explain|describe|what is|what are|who is|give me)\s+",
    re.IGNORECASE,
)
_MULTI_SPACE = re.compile(r"\s{2,}")


def _rewrite_query(query: str) -> str:
    """
    Lightweight rule-based query rewriting for improved retrieval.
    - Strips common filler phrases
    - Normalises whitespace
    - Expands common abbreviations
    """
    q = _FILLER.sub("", query).strip()
    q = _MULTI_SPACE.sub(" ", q)
    _ABBR = {
        r"\bIE\b": "information extraction",
        r"\bNLP\b": "natural language processing",
        r"\bML\b": "machine learning",
        r"\bAI\b": "artificial intelligence",
        r"\bRAG\b": "retrieval augmented generation",
        r"\bLLM\b": "large language model",
        r"\bAPI\b": "application programming interface",
        r"\bSD\b": "stable diffusion",
    }
    for pattern, expansion in _ABBR.items():
        q = re.sub(pattern, expansion, q)
    if q != query:
        logger.debug("[QUERY_REWRITE] '%s' → '%s'", query[:80], q[:80])
    return q or query


class QueryEngine:
    def __init__(
        self,
        model_loader: ModelLoader,
        hybrid_retriever: HybridRetriever,
        max_tokens: int = 384,
        temperature: float = 0.2,
        unload_on_memory_pressure: bool = True,
        ram_pressure_limit_pct: int = 85,
    ):
        self.loader = model_loader
        self.retriever = hybrid_retriever
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.unload_on_memory_pressure = unload_on_memory_pressure
        self.ram_pressure_limit_pct = ram_pressure_limit_pct
        self._lock = threading.Lock()
        self._sm = QueryStateMachine(
            {
                State.INIT: self._handle_init,
                State.LOAD: self._handle_load,
                State.RETRIEVE: self._handle_retrieve,
                State.FILTER: self._handle_filter,
                State.THINK: self._handle_think,
                State.ANSWER: self._handle_answer,
                State.CLEANUP: self._handle_cleanup,
                State.EXIT: self._handle_exit,
            }
        )

    def query(self, user_query: str) -> Dict[str, Any]:
        with self._lock:
            try:
                result = self._sm.run(user_query)
                result["ok"] = not bool(result.get("error"))
                return result
            except Exception as exc:
                classified = classify_error(exc)
                logger.exception("[QE] Unhandled error in query cycle.")
                return {
                    "ok": False,
                    "answer": "",
                    "error": str(exc),
                    **classified.to_dict(),
                }

    def _handle_init(self, ctx: dict) -> dict:
        query = str(ctx.get("query", "")).strip()
        if not query:
            raise ValueError("Query is empty.")
        if len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"Query exceeds {MAX_QUERY_CHARS} characters.")
        rewritten = _rewrite_query(query)
        ctx.update(
            {
                "query": query,
                "rewritten_query": rewritten,
                "chunks": [],
                "context_str": "",
                "answer": "",
                "error": None,
                "no_match": False,
            }
        )
        logger.info("[INIT] Query accepted (%d chars). Rewritten: '%s'",
                    len(query), rewritten[:80])
        return ctx

    def _handle_load(self, ctx: dict) -> dict:
        ctx["pre_snap"] = MemoryManager.snapshot()
        logger.info("[LOAD] Pre-query memory: %s", ctx["pre_snap"])
        return ctx

    def _handle_retrieve(self, ctx: dict) -> dict:
        effective_query = ctx.get("rewritten_query") or ctx["query"]
        results = self.retriever.retrieve(effective_query, use_vector=True)
        ctx["chunks"] = results
        ctx["context_str"] = self.retriever.format_context(results)
        logger.info("[RETRIEVE] %d chunk(s) retrieved for query '%s'.",
                    len(results), effective_query[:60])
        for idx, (chunk, score) in enumerate(results, 1):
            preview = chunk[:100].replace("\n", " ")
            logger.debug("[RETRIEVE] Chunk %d | score=%.4f | '%s...'", idx, score, preview)
        return ctx

    def _handle_filter(self, ctx: dict) -> dict:
        ctx["no_match"] = not bool(ctx["chunks"])
        if ctx["no_match"]:
            logger.info("[FILTER] No relevant context found in knowledge base.")
        else:
            scores = [s for _, s in ctx["chunks"]]
            logger.info("[FILTER] %d chunk(s) passed filter. Score range: %.3f–%.3f.",
                        len(ctx["chunks"]), min(scores), max(scores))
        return ctx

    def _handle_think(self, ctx: dict) -> dict:
        if ctx["no_match"]:
            ctx["final_prompt"] = None
            return ctx

        ctx["final_prompt"] = (
            "You are GlobAI, a precise offline RAG assistant.\n"
            "The retrieved context is the only source of truth. "
            "Do not use prior knowledge, guesses, or unstated assumptions. "
            "If the context does not directly answer the question, say exactly: "
            "'Insufficient data in knowledge base.'\n"
            "Keep the answer concise and do not invent citations.\n\n"
            f"Retrieved context:\n{ctx['context_str']}\n\n"
            f"User question:\n{ctx['query']}\n\n"
            "Answer:"
        )
        logger.info("[THINK] Prompt assembled (%d chars).", len(ctx["final_prompt"]))
        return ctx

    def _handle_answer(self, ctx: dict) -> dict:
        if ctx["no_match"]:
            ctx["answer"] = (
                "Insufficient data in knowledge base. "
                "Upload or index related documents, then ask again."
            )
            return ctx

        prompt = ctx.get("final_prompt")
        if not prompt:
            raise RuntimeError("Prompt assembly failed.")

        try:
            if not self.loader.is_loaded():
                logger.info("[ANSWER] Loading text model after context passed filters.")
                self.loader.load()
            answer = self.loader.generate(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            ctx["answer"] = answer or "Insufficient data in knowledge base."
            logger.info("[ANSWER] Generated %d chars.", len(ctx["answer"]))
        except Exception as exc:
            classified = classify_error(exc)
            ctx["answer"] = "Insufficient data in knowledge base."
            ctx["error"] = str(exc)
            ctx.update(classified.to_dict())
            logger.error("[ANSWER] Generation failed (%s): %s", classified.category, exc)
        return ctx

    def _handle_cleanup(self, ctx: dict) -> dict:
        ctx.pop("final_prompt", None)
        ctx.pop("context_str", None)
        self.loader.cleanup_after_generate()
        post_snap = MemoryManager.clear_cycle()
        ctx["post_snap"] = post_snap

        if self.unload_on_memory_pressure and post_snap.ram_used_pct >= self.ram_pressure_limit_pct:
            logger.warning(
                "[CLEANUP] RAM pressure %.1f%% >= %d%%; unloading LLM.",
                post_snap.ram_used_pct,
                self.ram_pressure_limit_pct,
            )
            self.loader.unload()
            MemoryManager.stabilize_after_unload("rag query pressure")
            ctx["model_unloaded_for_pressure"] = True

        logger.info("[CLEANUP] Cycle complete. Post-query memory: %s", post_snap)
        return ctx

    def _handle_exit(self, ctx: dict) -> dict:
        logger.info("[EXIT] Query cycle complete.")
        return ctx
