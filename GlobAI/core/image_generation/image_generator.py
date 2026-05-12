"""
core/image_generation/image_generator.py
----------------------------------------
Public image generation facade.
"""

from __future__ import annotations

import logging
from pathlib import Path

from core.image_generation.image_pipeline import IMAGE_MODEL_NOT_LOADED, ImagePipeline
from core.image_generation.sd_model_manager import SDModelManager

logger = logging.getLogger(__name__)


class ImageGenerator:
    def __init__(self, manager: SDModelManager):
        self.manager = manager
        self.pipeline = ImagePipeline(manager)

    def generate_image(
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
        try:
            return self.pipeline.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                output_dir=output_dir,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )
        except Exception as exc:
            logger.exception("[IMAGE_GENERATOR] Isolated image generation failure.")
            return {"ok": False, "error": str(exc)}
