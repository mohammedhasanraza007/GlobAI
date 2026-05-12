"""
core/image_generation/image_cache.py
------------------------------------
Local-only Stable Diffusion 1.5 single-file cache.
"""

from __future__ import annotations

import logging
from pathlib import Path

from bootstrap.model_cache import (
    SD15_LOCAL_FILENAME,
    normalize_model_id,
    sd_single_file_path,
)

logger = logging.getLogger(__name__)

SD15_SINGLE_FILE_LOAD_ERROR = "SD1.5 single-file model failed to load"

# The valid v1-5-pruned-emaonly.safetensors is ~4.27 GB.
# Anything under 4 GB is a corrupt or incomplete download.
SD15_MIN_FILE_SIZE_BYTES = 4_000_000_000


class ImageModelIntegrityError(RuntimeError):
    pass


class ImageCache:
    def __init__(
        self,
        cache_dir: str | Path = "model_checkpoints",
    ):
        self.cache_dir = Path(cache_dir)

    def normalize_model_id(self, model_id: str) -> str:
        return normalize_model_id(model_id)

    def target_path(self) -> Path:
        return sd_single_file_path(self.cache_dir)

    def validate_single_file(self, path: Path) -> Path:
        target = self.target_path().resolve()
        resolved = path.resolve()
        if resolved != target:
            raise ImageModelIntegrityError(SD15_SINGLE_FILE_LOAD_ERROR)
        if not resolved.is_file():
            raise ImageModelIntegrityError(
                f"{SD15_SINGLE_FILE_LOAD_ERROR}: file does not exist at {resolved}"
            )
        if resolved.name != SD15_LOCAL_FILENAME or resolved.suffix.lower() != ".safetensors":
            raise ImageModelIntegrityError(SD15_SINGLE_FILE_LOAD_ERROR)
        file_size = resolved.stat().st_size
        if file_size < SD15_MIN_FILE_SIZE_BYTES:
            size_gb = file_size / (1024 ** 3)
            logger.error(
                "[IMAGE_CACHE] SD1.5 file is too small (%.2f GB). "
                "Expected >= %.2f GB. File is corrupt or incomplete: %s",
                size_gb, SD15_MIN_FILE_SIZE_BYTES / (1024 ** 3), resolved,
            )
            raise ImageModelIntegrityError(
                f"{SD15_SINGLE_FILE_LOAD_ERROR}: file is only {size_gb:.2f} GB "
                f"(expected >= {SD15_MIN_FILE_SIZE_BYTES / (1024 ** 3):.2f} GB). "
                f"Delete {resolved} and re-run build.bat to re-download."
            )
        logger.info("[IMAGE_CACHE] Validated SD1.5 single-file checkpoint: %s (%.2f GB)", resolved, file_size / (1024 ** 3))
        return resolved

    def resolve_model_path(self, model_id: str = "sd1.5") -> Path:
        try:
            return self.validate_single_file(self.target_path())
        except ImageModelIntegrityError:
            logger.warning(
                "[IMAGE_CACHE] SD1.5 checkpoint must exist at exact path: %s",
                self.target_path(),
            )
        raise ImageModelIntegrityError(SD15_SINGLE_FILE_LOAD_ERROR)
