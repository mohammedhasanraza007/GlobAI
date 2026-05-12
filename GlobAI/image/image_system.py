"""
image/image_system.py
---------------------
Image generation owns only SD1.5 single-file loading and prompt-to-image output.
No RAG, embeddings, vector store, or coder objects are referenced here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.image_generation.image_generator import ImageGenerator
from core.image_generation.sd_model_manager import SDModelManager
from core.memory_manager import MemoryManager


class ImageSystem:
    name = "image"

    def __init__(self, config: dict[str, Any], cache_dir: str):
        self.config = config
        self.manager = SDModelManager(
            cache_dir=cache_dir,
            device=str(config.get("image_device", "cpu")),
        )
        self.generator = ImageGenerator(self.manager)
        self.generation_count = 0
        self.rag_access_count = 0
        self.coder_access_count = 0

    def load(self, model_id: str | None = None) -> dict[str, Any]:
        return {"ok": True, **self.manager.load_sd_model(model_id or self.config["image_model_id"])}

    def unload(self) -> None:
        if self.manager.image_model_loaded:
            self.manager.unload_sd_model()
        MemoryManager.hard_cleanup("image subsystem unload")
        MemoryManager.stabilize_after_unload("image subsystem")

    def generate(self, prompt: str, output_dir: str | Path = "generated/images", **kwargs: Any) -> dict[str, Any]:
        if not self.manager.image_model_loaded:
            self.load()
        self.generation_count += 1
        return self.generator.generate_image(prompt, output_dir=output_dir, **kwargs)

    def audit_state(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "image_loaded": self.manager.image_model_loaded,
            "generation_count": self.generation_count,
            "rag_access_count": self.rag_access_count,
            "coder_access_count": self.coder_access_count,
        }
