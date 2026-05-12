"""
core/image_generation/image_pipeline.py
---------------------------------------
Stable Diffusion generation wrapper with deterministic defaults.

Freeze protection:
- All pipeline calls are wrapped in a 60-second timeout via a background thread.
- If the primary device (DirectML / CUDA) times out or fails, generation is
  automatically retried on CPU using lighter settings.
- Resolution capped at 512×512; steps capped at 20–30.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import time
from pathlib import Path

import torch
from PIL import Image

from core.image_generation.sd_model_manager import SDModelManager
from core.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

IMAGE_MODEL_NOT_LOADED = "Image model not loaded. Load it first."
_GENERATION_TIMEOUT_SECS = 60
_CPU_GENERATION_TIMEOUT_SECS = 600
_CPU_FALLBACK_STEPS = 20


class ImagePipeline:
    def __init__(self, manager: SDModelManager):
        self.manager = manager

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _run_pipe(self, pipe, call_kwargs: dict):
        """Execute the pipeline synchronously. Runs inside a timeout thread."""
        return pipe(**call_kwargs)

    def _generate_on_device(self, pipe, call_kwargs: dict) -> object:
        """
        Run pipeline with a hard wall-clock timeout.
        Raises TimeoutError if generation does not complete in time.
        """
        timeout = (
            _CPU_GENERATION_TIMEOUT_SECS
            if self.manager._load_device_kind == "cpu"
            else _GENERATION_TIMEOUT_SECS
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._run_pipe, pipe, call_kwargs)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(
                    f"Image generation timed out after {timeout}s on {self.manager._load_device_kind}."
                )

    def _validate_saved_image(self, path: Path) -> None:
        try:
            with Image.open(path) as saved:
                saved.verify()
            with Image.open(path) as saved:
                if saved.width < 1 or saved.height < 1:
                    raise RuntimeError("Generated image has invalid dimensions.")
                
                from PIL import ImageStat
                stat = ImageStat.Stat(saved.convert("L"))
                stddev = stat.stddev[0]
                if stddev < 2.0:
                    raise RuntimeError(f"Generated image was essentially blank (stddev: {stddev:.2f}).")
                    
                extrema = saved.convert("L").getextrema()
                if not extrema or extrema[0] == extrema[1]:
                    raise RuntimeError("Generated image was completely uniform.")
        except Exception as exc:
            raise RuntimeError(f"Generated image failed validation: {exc}") from exc

    def _fallback_to_cpu(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int,
    ) -> object:
        """
        Reload model on CPU and regenerate with lighter settings.
        Called when the primary device (DirectML) times out or stalls.
        """
        model_path = self.manager._model_path
        if model_path is None:
            raise RuntimeError("No model path available for CPU fallback.")

        logger.warning(
            "[IMAGE_PIPELINE] Primary device (%s) timed out. "
            "Falling back to CPU generation.",
            self.manager._load_device_kind,
        )

        from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

        cpu_pipe = StableDiffusionPipeline.from_single_file(
            str(model_path),
            local_files_only=True,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )
        cpu_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            cpu_pipe.scheduler.config
        )
        cpu_pipe.enable_attention_slicing()
        cpu_pipe.set_progress_bar_config(disable=True)
        cpu_pipe = cpu_pipe.to("cpu")

        generator = torch.Generator(device="cpu").manual_seed(seed)
        call_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            width=min(width, 512),
            height=min(height, 512),
            num_inference_steps=_CPU_FALLBACK_STEPS,
            guidance_scale=float(guidance_scale),
            generator=generator,
        )

        try:
            result = cpu_pipe(**call_kwargs)
        finally:
            del cpu_pipe
            MemoryManager.hard_cleanup("cpu fallback generation")

        logger.info("[IMAGE_PIPELINE] CPU fallback generation completed.")
        return result

    # ── Public interface ───────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        output_dir: str | Path = "generated/images",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        guidance_scale: float = 7.5,
        seed: int = 0,
    ) -> dict[str, object]:
        if not self.manager.image_model_loaded:
            return {"ok": False, "error": IMAGE_MODEL_NOT_LOADED}

        prompt = str(prompt or "").strip()
        negative_prompt = str(negative_prompt or "").strip()
        if not prompt:
            return {"ok": False, "error": "Image prompt is empty."}

        # Hard limits: 512×512, 20–30 steps
        width = max(64, min(512, int(width)))
        height = max(64, min(512, int(height)))
        steps = max(20, min(30, int(steps)))
        seed = int(seed)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "[IMAGE_PIPELINE] Generation start. Device: %s | %dx%d | %d steps. Memory: %s",
            self.manager._load_device_kind, width, height, steps,
            MemoryManager.snapshot(),
        )

        image = None
        tmp_path: Path | None = None
        try:
            generator_device = (
                "cuda" if self.manager._load_device_kind == "cuda" else "cpu"
            )
            generator = torch.Generator(device=generator_device).manual_seed(seed)
            call_kwargs = dict(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=float(guidance_scale),
                generator=generator,
            )

            try:
                result = self._generate_on_device(self.manager.pipeline, call_kwargs)
            except TimeoutError as tex:
                logger.warning("[IMAGE_PIPELINE] %s", tex)
                if self.manager._load_device_kind == "cpu":
                    raise
                result = self._fallback_to_cpu(
                    prompt, negative_prompt, width, height, steps, guidance_scale, seed
                )

            image = result.images[0]
            filename = f"sd_{int(time.time())}_{abs(seed)}.png"
            final_path = output_path / filename
            tmp_path = output_path / f"{final_path.stem}.tmp.png"
            image.save(tmp_path)
            self._validate_saved_image(tmp_path)
            os.replace(tmp_path, final_path)
            logger.info("[IMAGE_PIPELINE] Generation complete: %s", final_path)
            return {"ok": True, "path": str(final_path), "seed": seed}

        except Exception as exc:
            logger.exception("[IMAGE_PIPELINE] Generation failed.")
            return {"ok": False, "error": str(exc)}
        finally:
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            if image is not None:
                del image
            MemoryManager.hard_cleanup("image generation cycle")
