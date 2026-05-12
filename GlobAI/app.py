"""
app.py
------
GlobAI composition root.

The app is wired as a bounded local pipeline:
input -> intent classifier -> retrieval / code gen / image -> cleanup.

V2 additions:
- Intent classifier for deterministic auto-routing
- coder_max_tokens, coder_generation_timeout config keys
- auto_route flag (default: True) for RAG-mode auto-dispatch
- Error classifier integration in route()
- stabilize_after_unload called on safe-mode entry
- startup is local-only; missing models are reported, not downloaded
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml



PROJECT_ROOT = Path(__file__).resolve().parent

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
LOG_DIR = PROJECT_ROOT / "logs"


APP_NAME = "GlobAI"
MODE_RAG = "RAG"
MODE_CODER = "CODER"
MODE_IMAGE = "IMAGE"
ACTIVE_MODE = MODE_RAG
VALID_MODES = {MODE_RAG, MODE_CODER, MODE_IMAGE}


DEFAULT_CONFIG: dict[str, Any] = {
    "offline_mode": True,
    "device": "auto",
    "llm_device": "auto",
    "embedding_device": "cpu",
    "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "coder_model_id": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "embedding_id": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_dimension": 384,
    "cache_dir": "model_checkpoints",
    "image_model_id": "sd1.5",
    "image_device": "cpu",
    "vector_db": "data/vector_db",
    "max_tokens": 384,
    "coder_max_tokens": 384,
    "coder_generation_timeout": 45,
    "max_input_tokens": 1536,
    "temperature": 0.0,
    "keep_model_loaded": True,
    "unload_on_memory_pressure": True,
    "ram_pressure_limit_pct": 80,
    "auto_route": True,
    "similarity_threshold": 0.24,
    "top_k": 3,
    "candidate_k": 10,
    "max_context_chars": 6000,
    "max_vector_budget": 5000,
    "dedupe_similarity_threshold": 0.97,
    "vector_dedupe_window": 128,
    "embedding_batch_size": 16,
    "chunk_size": 220,
    "chunk_overlap": 40,
    "max_document_chars": 2_000_000,
    "max_file_bytes": 25_000_000,
    "max_pdf_pages": 250,
    "max_pptx_slides": 250,
    "max_docx_paragraphs": 5000,
    "system_prompt": (
        "You are a precise, deterministic AI. "
        "Follow instructions exactly. Do not hallucinate."
    ),
}


def _setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()

    if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers):
        handler = logging.handlers.RotatingFileHandler(
            LOG_DIR / "globai.log",
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        root.addHandler(handler)

    if not any(type(h) is logging.StreamHandler for h in root.handlers):
        root.addHandler(logging.StreamHandler(sys.stdout))

    root.setLevel(logging.INFO)


def _bounded_int(cfg: dict[str, Any], key: str, low: int, high: int) -> int:
    default = DEFAULT_CONFIG.get(key, low)
    value = int(cfg.get(key, default))
    return max(low, min(high, value))


def _bounded_float(cfg: dict[str, Any], key: str, low: float, high: float) -> float:
    default = DEFAULT_CONFIG.get(key, low)
    value = float(cfg.get(key, default))
    return max(low, min(high, value))


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_config(raw: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(raw or {})

    for key in (
        "offline_mode",
        "keep_model_loaded",
        "unload_on_memory_pressure",
        "auto_route",
    ):
        cfg[key] = _coerce_bool(cfg.get(key, DEFAULT_CONFIG.get(key, False)))

    cfg["max_tokens"] = _bounded_int(cfg, "max_tokens", 32, 768)
    cfg["coder_max_tokens"] = _bounded_int(cfg, "coder_max_tokens", 64, 1024)
    cfg["coder_generation_timeout"] = _bounded_int(cfg, "coder_generation_timeout", 10, 300)
    cfg["max_input_tokens"] = _bounded_int(cfg, "max_input_tokens", 256, 4096)
    cfg["temperature"] = _bounded_float(cfg, "temperature", 0.0, 1.0)
    cfg["ram_pressure_limit_pct"] = _bounded_int(cfg, "ram_pressure_limit_pct", 50, 95)
    cfg["top_k"] = _bounded_int(cfg, "top_k", 1, 6)
    cfg["candidate_k"] = max(cfg["top_k"], _bounded_int(cfg, "candidate_k", 1, 24))
    cfg["max_context_chars"] = _bounded_int(cfg, "max_context_chars", 500, 12_000)
    cfg["max_vector_budget"] = _bounded_int(cfg, "max_vector_budget", 100, 20_000)
    cfg["dedupe_similarity_threshold"] = _bounded_float(
        cfg, "dedupe_similarity_threshold", 0.80, 0.999
    )
    cfg["vector_dedupe_window"] = _bounded_int(cfg, "vector_dedupe_window", 0, 2048)
    cfg["embedding_batch_size"] = _bounded_int(cfg, "embedding_batch_size", 1, 64)
    cfg["embedding_dimension"] = _bounded_int(cfg, "embedding_dimension", 1, 4096)
    cfg["chunk_size"] = _bounded_int(cfg, "chunk_size", 50, 800)
    cfg["chunk_overlap"] = _bounded_int(cfg, "chunk_overlap", 0, cfg["chunk_size"] - 1)
    cfg["max_document_chars"] = _bounded_int(cfg, "max_document_chars", 10_000, 10_000_000)
    cfg["max_file_bytes"] = _bounded_int(cfg, "max_file_bytes", 100_000, 100_000_000)
    cfg["max_pdf_pages"] = _bounded_int(cfg, "max_pdf_pages", 1, 1000)
    cfg["max_pptx_slides"] = _bounded_int(cfg, "max_pptx_slides", 1, 1000)
    cfg["max_docx_paragraphs"] = _bounded_int(cfg, "max_docx_paragraphs", 1, 20_000)
    return cfg


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        with CONFIG_PATH.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(DEFAULT_CONFIG, fh, sort_keys=False)
        return dict(DEFAULT_CONFIG)

    with CONFIG_PATH.open(encoding="utf-8") as fh:
        return _coerce_config(yaml.safe_load(fh) or {})


def _configure_offline_mode(enabled: bool) -> None:
    if not enabled:
        return
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def normalize_mode(mode: str) -> str:
    normalized = str(mode or "").strip().upper()
    aliases = {
        "RAG": MODE_RAG,
        "TEXT": MODE_RAG,
        "CODER": MODE_CODER,
        "CODE": MODE_CODER,
        "IMAGE": MODE_IMAGE,
        "IMG": MODE_IMAGE,
    }
    if normalized not in aliases:
        raise ValueError(f"Unknown mode: {mode}")
    return aliases[normalized]


@dataclass
class App:
    config: dict[str, Any]
    rag_system: Any
    coder_system: Any
    image_system: Any
    mode: str

    @property
    def query_engine(self) -> Any:
        return self.rag_system.query_engine

    @property
    def embedding_engine(self) -> Any:
        return self.rag_system.embedding_engine

    @property
    def vector_store(self) -> Any:
        return self.rag_system.vector_store

    @property
    def bm25_retriever(self) -> Any:
        return self.rag_system.bm25_retriever

    @property
    def hybrid_retriever(self) -> Any:
        return self.rag_system.hybrid_retriever

    @property
    def model_loader(self) -> Any:
        return self.rag_system.model_loader

    @property
    def chunker(self) -> Any:
        return self.rag_system.chunker

    @property
    def image_model_manager(self) -> Any:
        return self.image_system.manager

    @property
    def image_generator(self) -> Any:
        return self.image_system.generator

    def _unload_every_runtime(self, reason: str) -> None:
        from core.memory_manager import MemoryManager

        self.rag_system.unload()
        self.coder_system.disable()
        self.image_system.unload()
        MemoryManager.stabilize_after_unload(reason)
        MemoryManager.enforce_ram_ceiling("post-unload stabilization")

    def _load_mode_runtime(self, target: str) -> dict[str, Any]:
        if target == MODE_RAG:
            self.rag_system.load()
            return {"ok": True, "mode": MODE_RAG, "model_id": self.config["model_id"]}
        if target == MODE_CODER:
            result = self.coder_system.enable()
            return {**result, "mode": MODE_CODER}
        if target == MODE_IMAGE:
            return {**self.image_system.load(), "mode": MODE_IMAGE}
        raise ValueError(f"Unknown mode: {target}")

    def switch_mode(self, target: str) -> dict[str, Any]:
        target_mode = normalize_mode(target)
        if self.mode == target_mode:
            return {"ok": True, "mode": self.mode, "status": "already_active"}

        previous_mode = self.mode
        self._unload_every_runtime(f"switch {previous_mode} to {target_mode}")
        try:
            result = self._load_mode_runtime(target_mode)
        except Exception as exc:
            from core.error_classifier import classify_error
            classified = classify_error(exc)
            self._unload_every_runtime(f"failed switch to {target_mode}")
            self.mode = ACTIVE_MODE
            return {
                "ok": False,
                "mode": self.mode,
                "error": str(exc),
                **classified.to_dict(),
            }

        self.mode = target_mode
        return {"ok": bool(result.get("ok", True)), **result, "previous_mode": previous_mode}

    def enter_safe_mode(self, reason: str) -> None:
        logger = logging.getLogger(__name__)
        logger.error("[APP] Entering RAG-only safe mode: %s", reason)
        self.coder_system.disable()
        self.image_system.unload()
        self.mode = ACTIVE_MODE
        try:
            from core.memory_manager import MemoryManager
            MemoryManager.stabilize_after_unload("safe mode entry")
        except Exception:
            pass

    def load_text_model(self) -> dict[str, Any]:
        if self.mode != MODE_RAG:
            return self.switch_mode(MODE_RAG)
        if self.rag_system.model_loader.is_loaded():
            return {"ok": True, "mode": MODE_RAG, "model_id": self.config["model_id"]}
        self.rag_system.load()
        return {"ok": True, "mode": MODE_RAG, "model_id": self.config["model_id"]}

    def unload_text_model(self) -> dict[str, Any]:
        self.rag_system.unload()
        return {"ok": True, "mode": self.mode}

    def enable_coder_mode(self) -> dict[str, Any]:
        return self.switch_mode(MODE_CODER)

    def disable_coder_mode(self) -> dict[str, Any]:
        self.coder_system.disable()
        self.mode = MODE_RAG
        return {"ok": True, "mode": self.mode}

    def run_coder(self, prompt: str) -> dict[str, Any]:
        if self.mode != MODE_CODER:
            return {"ok": False, "error": "Coder mode not active. Enable it first."}
        if not self.coder_system.enabled or not self.coder_system.loader.is_loaded():
            enabled = self.coder_system.enable()
            if not enabled.get("ok"):
                return enabled
        result = self.coder_system.run(prompt)
        
        # Force unload to aggressively save RAM
        self.coder_system.unload()
        
        self.mode = MODE_CODER
        return {**result, "mode": self.mode}

    def load_sd_model(self, model_id: str | None = None) -> dict[str, Any]:
        if self.mode != MODE_IMAGE:
            switched = self.switch_mode(MODE_IMAGE)
            if not switched.get("ok") or not model_id:
                return switched
            self.image_system.unload()
        if self.image_system.manager.image_model_loaded and not model_id:
            return {"ok": True, "mode": MODE_IMAGE, "status": "already_loaded"}
        return {**self.image_system.load(model_id), "mode": MODE_IMAGE}

    def unload_sd_model(self) -> dict[str, Any]:
        self.image_system.unload()
        self.mode = MODE_RAG
        return {"ok": True, "status": "unloaded"}

    def _enforce_image_result(self, result: dict[str, Any]) -> dict[str, Any]:
        text_keys = ("answer", "text", "content", "message")
        if any(str(result.get(key) or "").strip() for key in text_keys):
            return {
                "ok": False,
                "mode": MODE_IMAGE,
                "error": "Image mode returned text instead of an image.",
            }
        if result.get("ok") and not result.get("path"):
            return {
                "ok": False,
                "mode": MODE_IMAGE,
                "error": "Image mode completed without an image path.",
            }
        return {**result, "mode": MODE_IMAGE}

    def generate_image(
        self,
        positive_prompt: str,
        negative_prompt: str = "",
        image_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.mode != MODE_IMAGE:
            switched = self.switch_mode(MODE_IMAGE)
            if not switched.get("ok"):
                return switched
        kwargs = dict(image_kwargs or {})
        kwargs["negative_prompt"] = negative_prompt
        result = self.image_system.generate(positive_prompt, **kwargs)
        
        # Force unload image model after generation
        self.image_system.unload()
        
        return self._enforce_image_result(result)

    def _auto_route(self, user_query: str, explicit_image: bool) -> dict[str, Any] | None:
        """
        V2 intent-based auto-routing when mode is RAG and auto_route is enabled.
        Routes coder/image queries transparently without permanently switching modes.
        Returns a result dict if routed, or None to fall through to RAG.
        """
        if not _coerce_bool(self.config.get("auto_route", True)):
            return None
        if self.mode != MODE_RAG or explicit_image:
            return None

        from core.intent_classifier import classify
        intent = classify(user_query)
        logger = logging.getLogger(__name__)

        if intent == "image":
            logger.info("[ROUTE] Auto-routing to IMAGE (classifier).")
            self.rag_system.unload() # Strict RAM: Unload RAG before loading SD
            kwargs: dict[str, Any] = {}
            result = self.image_system.generate(user_query, **kwargs)
            # image_system.generate calls load() if needed, and we assume it unloads if it's supposed to
            # But generate_image in app.py unloads it. Let's make sure it's unloaded.
            self.image_system.unload()
            return self._enforce_image_result(result)

        if intent == "coder":
            logger.info("[ROUTE] Auto-routing to CODER (classifier).")
            self.rag_system.unload() # Strict RAM: Unload RAG before loading Coder
            self.coder_system.enabled = True
            result = self.coder_system.run(user_query)
            
            # Force unload after auto-routing coder
            self.coder_system.unload()
            
            return {**result, "mode": MODE_CODER, "auto_routed": True}

        return None

    # ── System prompt control ──────────────────────────────────────────────────

    def get_system_prompt(self) -> str:
        """Return the active system prompt."""
        from core.system_prompt import get_system_prompt
        return get_system_prompt()

    def set_system_prompt(self, new_prompt: str) -> None:
        """Update the active system prompt in memory (runtime only)."""
        from core.system_prompt import set_system_prompt
        set_system_prompt(new_prompt)
        self.config["system_prompt"] = new_prompt

    def save_system_prompt(self, new_prompt: str) -> None:
        """
        Persist the system prompt to config.yaml.
        Only rewrites the system_prompt line; all other lines and comments are preserved.
        """
        import re
        self.set_system_prompt(new_prompt)
        escaped = new_prompt.replace("\\", "\\\\").replace('"', '\\"')
        replacement = f'system_prompt: "{escaped}"'
        content = CONFIG_PATH.read_text(encoding="utf-8")
        if re.search(r"^system_prompt:", content, re.MULTILINE):
            content = re.sub(
                r"^system_prompt:.*$", replacement, content, flags=re.MULTILINE
            )
        else:
            content = content.rstrip("\n") + f"\n{replacement}\n"
        CONFIG_PATH.write_text(content, encoding="utf-8")

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index_documents(self, paths: list[str | Path]) -> dict[str, Any]:
        """Index documents into the RAG system."""
        try:
            count = self.rag_system.index_paths(paths)
            return {"ok": True, "count": count}
        except Exception as exc:
            from core.error_classifier import classify_error
            classified = classify_error(exc)
            return {
                "ok": False,
                "error": str(exc),
                **classified.to_dict(),
            }

    # ── Routing ────────────────────────────────────────────────────────────────

    def route(
        self,
        user_query: str,
        explicit_image_request: bool = False,
        image_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        query_text = str(user_query or "")
        try:
            if self.mode == MODE_IMAGE or explicit_image_request:
                kwargs = dict(image_kwargs or {})
                negative_prompt = str(kwargs.pop("negative_prompt", ""))
                return self.generate_image(
                    positive_prompt=query_text,
                    negative_prompt=negative_prompt,
                    image_kwargs=kwargs,
                )
            if self.mode == MODE_CODER:
                return self.run_coder(query_text)

            auto_result = self._auto_route(query_text, explicit_image_request)
            if auto_result is not None:
                return auto_result

            if self.mode != MODE_RAG:
                switched = self.switch_mode(MODE_RAG)
                if not switched.get("ok"):
                    return switched
            return self.rag_system.query(query_text)
        except Exception as exc:
            from core.error_classifier import classify_error
            classified = classify_error(exc)
            self.enter_safe_mode(str(exc))
            return {
                "ok": False,
                "error": str(exc),
                "mode": self.mode,
                **classified.to_dict(),
            }

    def audit_isolation(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "rag": self.rag_system.audit_state(),
            "coder": self.coder_system.audit_state(),
            "image": self.image_system.audit_state(),
            "shared_loader_objects": self.rag_system.model_loader is self.coder_system.loader,
            "shared_image_rag_objects": self.image_system.manager is self.rag_system.model_loader,
        }


def build_app(skip_preflight: bool = False) -> App:
    _setup_logging()
    logger = logging.getLogger(__name__)

    if not skip_preflight:
        from bootstrap.preflight_check import run_preflight

        run_preflight()

    cfg = load_config()

    from core.system_prompt import set_system_prompt as _set_sp
    _set_sp(str(cfg.get("system_prompt", "")))
    logger.info("[APP] System prompt loaded (%d chars).", len(cfg.get("system_prompt", "")))

    from bootstrap.model_cache import prepare_model_cache, resolve_cache_dir

    cache_status = prepare_model_cache(PROJECT_ROOT, cfg)
    if cache_status.get("missing"):
        logger.warning("[APP] Model cache missing local files: %s", cache_status["missing"])
    _configure_offline_mode(bool(cfg.get("offline_mode", False)))

    cache_dir = str(resolve_cache_dir(PROJECT_ROOT, cfg["cache_dir"]))
    vector_db_path = str(PROJECT_ROOT / cfg["vector_db"])

    from core.device_resolver import log_device_banner, resolve_device

    llm_pref = str(cfg.get("llm_device") or cfg.get("device") or "cpu")
    embed_pref = str(cfg.get("embedding_device") or "cpu")

    llm_device, llm_kind = resolve_device(llm_pref)
    embed_device, embed_kind = resolve_device(embed_pref)
    log_device_banner(llm_device, llm_kind, label="LLM")
    if embed_kind != llm_kind:
        log_device_banner(embed_device, embed_kind, label="Embedding")

    from coder.coder_system import CoderSystem
    from image.image_system import ImageSystem
    from rag.rag_system import RagSystem

    rag_system = RagSystem(
        config=cfg,
        cache_dir=cache_dir,
        vector_db_path=vector_db_path,
        llm_device_preference=llm_pref,
        embed_device=embed_device,
        embed_device_kind=embed_kind,
    )
    coder_system = CoderSystem(config=cfg, cache_dir=cache_dir, device_preference=llm_pref)
    image_system = ImageSystem(config=cfg, cache_dir=cache_dir)

    logger.info("[APP] %s V2 ready. Coder model: %s", APP_NAME, cfg.get("coder_model_id"))
    return App(
        config=cfg,
        rag_system=rag_system,
        coder_system=coder_system,
        image_system=image_system,
        mode=ACTIVE_MODE,
    )
