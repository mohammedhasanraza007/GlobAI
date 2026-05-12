"""
Validate that all configured GlobAI model artifacts exist in the local cache.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yaml

from bootstrap.model_cache import (
    prepare_model_cache,
    required_models_from_config,
    resolve_cache_dir,
    resolve_required_model,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


def main() -> int:
    cfg = _load_config()
    cache_dir = resolve_cache_dir(PROJECT_ROOT, cfg.get("cache_dir", "model_checkpoints"))
    prepare_model_cache(PROJECT_ROOT, cfg)

    missing: list[str] = []
    for item in required_models_from_config(cfg):
        found = resolve_required_model(cache_dir, item.repo_id, item.role)
        if found is None:
            missing.append(f"{item.role}:{item.repo_id}")
        else:
            print(f"[VALIDATE] {item.role}: {found}")

    if missing:
        print("[VALIDATE] Missing required model artifact(s):")
        for item in missing:
            print(f"  - {item}")
        return 1

    print("[VALIDATE] All required models are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
