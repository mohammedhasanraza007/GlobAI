"""
core/image_generation/sd_model_manager.py
-----------------------------------------
Explicit local Stable Diffusion lifecycle manager.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import torch

from core.image_generation.image_cache import ImageCache, SD15_SINGLE_FILE_LOAD_ERROR
from core.memory_manager import (
    DuplicateModelLoadError,
    MemoryManager,
    ModelCleanupError,
)

logger = logging.getLogger(__name__)


class SDModelManager:
    def __init__(
        self,
        cache_dir: str | Path = "model_checkpoints",
        device: str = "cpu",
    ):
        self.cache = ImageCache(cache_dir=cache_dir)
        self.device = str(device or "cpu")
        self._pipeline: Any = None
        self._model_id: str | None = None
        self._model_path: Path | None = None
        self._load_device: Any = None
        self._load_device_kind: str = "cpu"
        self._lock = threading.RLock()

    @property
    def image_model_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def pipeline(self) -> Any:
        return self._pipeline

    def _select_load_device(self) -> tuple[Any, str]:
        preference = self.device.strip().lower()
        if preference == "cpu":
            return torch.device("cpu"), "cpu"

        if preference == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda"), "cuda"
            logger.warning("[IMAGE] CUDA requested but unavailable; using CPU.")
            return torch.device("cpu"), "cpu"

        if preference == "directml":
            try:
                import torch_directml  # type: ignore

                return torch_directml.device(), "directml"
            except Exception as exc:
                logger.warning("[IMAGE] DirectML requested but unavailable (%s); using CPU.", exc)
                return torch.device("cpu"), "cpu"

        if torch.cuda.is_available():
            logger.info("[IMAGE] Using CUDA device.")
            return torch.device("cuda"), "cuda"
        try:
            import torch_directml  # type: ignore

            logger.info("[IMAGE] Using DirectML device.")
            return torch_directml.device(), "directml"
        except Exception as exc:
            logger.info("[IMAGE] DirectML unavailable (%s); using CPU.", exc)
            return torch.device("cpu"), "cpu"

    def load_sd_model(self, model_id: str = "sd1.5") -> dict[str, str]:
        with self._lock:
            normalized = self.cache.normalize_model_id(model_id)
            if self.image_model_loaded and self._model_id == normalized:
                raise DuplicateModelLoadError(f"Model already loaded: {normalized}")
            if self.image_model_loaded:
                self.unload_sd_model()

            pipe = None
            try:
                model_path = self.cache.resolve_model_path(normalized)
                before = MemoryManager.snapshot()

                # Respect image_device; auto probes CUDA, then DirectML, then CPU.
                load_device, device_kind = self._select_load_device()

                logger.info(
                    "[IMAGE] Loading SD1.5 single-file model %s from %s. Device: %s. Before: %s",
                    normalized,
                    model_path,
                    device_kind,
                    before,
                )
                from diffusers import StableDiffusionPipeline

                dtype = torch.float16 if device_kind == "cuda" else torch.float32
                pipe = StableDiffusionPipeline.from_single_file(
                    str(model_path),
                    local_files_only=True,
                    torch_dtype=dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                )
                pipe = pipe.to(load_device)
                pipe.enable_attention_slicing()
                pipe.set_progress_bar_config(disable=True)
                # Use DPMSolverMultistepScheduler for all devices:
                # fast (converges in 20 steps), compatible with CPU and DirectML,
                # and avoids the heavy schedulers that stall on non-CUDA backends.
                from diffusers import DPMSolverMultistepScheduler
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config
                )

                self._pipeline = pipe
                self._model_id = normalized
                self._model_path = model_path
                self._load_device = load_device
                self._load_device_kind = device_kind
                logger.info("[IMAGE] SD1.5 single-file model loaded. After: %s", MemoryManager.snapshot())
                return {"status": "loaded", "model_id": normalized, "path": str(model_path)}
            except Exception:
                logger.exception("[IMAGE] SD1.5 single-file load failed.")
                # Aggressively clear all partial state
                if pipe is not None:
                    try:
                        # Clear sub-components that may hold large tensors
                        for attr in ("unet", "vae", "text_model", "tokenizer", "scheduler", "safety_checker"):
                            if hasattr(pipe, attr):
                                setattr(pipe, attr, None)
                    except Exception:
                        pass
                    del pipe
                    pipe = None
                self._pipeline = None
                self._model_id = None
                self._model_path = None
                self._load_device = None
                self._load_device_kind = "cpu"
                MemoryManager.hard_cleanup("failed image model load")
                MemoryManager.emergency_reclaim("failed image model load")
                raise RuntimeError(SD15_SINGLE_FILE_LOAD_ERROR)

    def unload_sd_model(self) -> dict[str, str]:
        with self._lock:
            before = MemoryManager.snapshot()
            logger.info("[IMAGE] Unload requested for %s. Before: %s", self._model_id, before)
            had_model = self._pipeline is not None
            pipe = self._pipeline
            model_id = self._model_id
            self._pipeline = None
            self._model_id = None
            self._model_path = None
            self._load_device = None
            self._load_device_kind = "cpu"
            if pipe is not None:
                del pipe
            after = MemoryManager.hard_cleanup("image model unload")
            if had_model and MemoryManager.memory_increased(before, after):
                logger.warning("[IMAGE] Memory increased after unload; running emergency cleanup.")
                after = MemoryManager.hard_cleanup("image emergency cleanup")
                if MemoryManager.memory_increased(before, after):
                    raise ModelCleanupError("Memory did not return to a safe level after image model unload.")
            logger.info("[IMAGE] SD model unloaded. After: %s", after)
            return {"status": "unloaded", "model_id": str(model_id or "")}
