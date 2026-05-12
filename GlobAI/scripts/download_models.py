"""
scripts/download_models.py
--------------------------
Download/repair all required local models into the role-aware cache layout.
"""

from __future__ import annotations

import os
import sys

# Ensure symlinks are disabled on Windows to prevent permission errors
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yaml

from bootstrap.model_cache import (
    create_cache_layout,
    download_model,
    migrate_legacy_cache,
    required_models_from_config,
    resolve_cache_dir,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


def main() -> int:
    # Ensure symlinks are disabled on Windows to prevent permission errors
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    
    cfg = _load_config()
    cache_dir = resolve_cache_dir(PROJECT_ROOT, cfg.get("cache_dir", "model_checkpoints"))
    required = required_models_from_config(cfg)

    create_cache_layout(cache_dir)
    migrate_legacy_cache(cache_dir, required)

    failures: list[str] = []
    for item in required:
        print(f"\n[DOWNLOAD] {item.role}: {item.repo_id}")
        try:
            path = download_model(item.repo_id, cache_dir, item.role)
            print(f"[DOWNLOAD] Ready: {path}")
        except Exception as exc:
            print(f"[DOWNLOAD] Failed: {exc}")
            failures.append(f"{item.role}:{item.repo_id}")

    if failures:
        print("\n[DOWNLOAD] Failed model(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\n[DOWNLOAD] All model files downloaded into role-specific cache folders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
