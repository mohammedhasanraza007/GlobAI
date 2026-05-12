"""
coder/coder_system.py
---------------------
Stable 0.5B-only coder system.

Behavior:
- one active coder model only
- no automatic fallback or model switching
- one deterministic generation per request
- bounded generation time through transformers max_time
"""

from __future__ import annotations

import gc
import logging
import threading
import ast
from typing import Any

from core.error_classifier import classify_error
from core.memory_manager import MemoryManager
from core.model_loader import ModelLoader

logger = logging.getLogger(__name__)

CODER_MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
CODER_SUPPORTED_MODELS = (CODER_MODEL_ID,)
CODER_INACTIVE_ERROR = "Coder mode not active."

_STOP_TOKENS = ("<|im_end|>", "<|endoftext|>", "</s>")


class CoderSystem:
    name = "coder"

    def __init__(
        self,
        config: dict[str, Any],
        cache_dir: str,
        device_preference: str,
    ):
        self.config = config
        self.config["coder_model_id"] = CODER_MODEL_ID

        self.loader = ModelLoader(
            model_id=CODER_MODEL_ID,
            device_preference=device_preference,
            cache_dir=cache_dir,
            keep_loaded=False,
            max_input_tokens=int(config.get("max_input_tokens", 1536)),
            local_files_only=True,
            model_role="coder",
        )

        self.enabled = False
        self.run_count = 0
        self.rag_access_count = 0
        self._lock = threading.Lock()

    def enable(self) -> dict[str, Any]:
        try:
            if self.loader.model_id != CODER_MODEL_ID:
                self.loader.switch_model(CODER_MODEL_ID, model_role="coder")
            if not self.loader.is_loaded():
                self.loader.load()

            self.enabled = True
            logger.info("[CODER] Enabled with %s", self.loader.model_id)
            return {"ok": True, "mode": self.name, "model_id": self.loader.model_id}
        except Exception as exc:
            logger.exception("[CODER] Load failed")
            self.enabled = False
            return {"ok": False, "error": str(exc)}

    def disable(self) -> dict[str, Any]:
        self.unload()
        self.enabled = False
        return {"ok": True}

    def unload(self) -> None:
        try:
            if self.loader.is_loaded():
                self.loader.unload()
        except Exception:
            logger.exception("[CODER] Unload failed")

        try:
            MemoryManager.hard_cleanup("coder unload")
            MemoryManager.stabilize_after_unload("coder")
        except Exception:
            pass

    def _format_prompt(self, task: str) -> str:
        task = str(task or "").strip()
        return (
            "<|im_start|>system\n"
            "You are GlobAI Coder. Write deterministic, runnable Python 3.10 code. "
            "Use only the standard library unless the user explicitly asks otherwise.\n"
            "CRITICAL: RETURN ONLY RAW VALID PYTHON CODE. NO MARKDOWN. NO EXPLANATIONS. "
            "If requested to create a game or graphical app (like ping pong), provide complete, runnable GUI code (e.g. using tkinter, pygame, or PyQt6).\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{task}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "```python\n"
        )

    def _strip_stop_tokens(self, text: str) -> str:
        clean = str(text or "")
        stop_positions = [clean.find(token) for token in _STOP_TOKENS if token in clean]
        if stop_positions:
            clean = clean[: min(stop_positions)]
        return clean.strip()

    def _extract_code(self, raw: str) -> str:
        raw = self._strip_stop_tokens(raw)

        if "```python" in raw:
            start = raw.find("```python") + len("```python")
            end = raw.find("```", start)
            if end != -1:
                return self._strip_stop_tokens(raw[start:end])
            return self._strip_stop_tokens(raw[start:])

        if "```" in raw:
            start = raw.find("```") + 3
            end = raw.find("```", start)
            if end != -1:
                return self._strip_stop_tokens(raw[start:end])

        return raw

    def run(self, prompt: str) -> dict[str, Any]:
        with self._lock:
            if not self.enabled:
                result = self.enable()
                if not result.get("ok"):
                    return result

            try:
                if not self.loader.is_loaded():
                    self.loader.load()

                raw = self.loader.generate(
                    self._format_prompt(prompt),
                    max_tokens=int(self.config.get("coder_max_tokens", 384)),
                    temperature=0.0,
                    max_time=float(self.config.get("coder_generation_timeout", 45)),
                )
                code = self._extract_code(raw)
                if not code:
                    return {"ok": False, "error": "Coder produced empty output."}

                try:
                    ast.parse(code)
                except SyntaxError as syntax_exc:
                    logger.warning("[CODER] Syntax validation failed: %s", syntax_exc)
                    # We still return the code as requested, but we could add a note.

                self.run_count += 1
                return {"ok": True, "answer": code, "model_id": self.loader.model_id}
            except Exception as exc:
                classified = classify_error(exc)
                logger.exception("[CODER] Generation failed: %s", exc)
                return {
                    "ok": False,
                    "error": str(exc),
                    "category": classified.category,
                }
            finally:
                gc.collect()
                try:
                    self.loader.cleanup_after_generate()
                except Exception:
                    pass

    def audit_state(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_id": self.loader.model_id,
            "coder_loaded": self.loader.is_loaded(),
            "run_count": self.run_count,
            "rag_access_count": self.rag_access_count,
        }
