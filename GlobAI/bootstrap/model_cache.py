"""
bootstrap/model_cache.py
------------------------
Role-aware local model cache management.

The cache layout is deliberately explicit:

model_checkpoints/
  embeddings/
  rag/
  coder/
  sd/

Legacy flat Hugging Face cache folders under model_checkpoints/models--...
are migrated into the correct role bucket on startup when possible.
Startup is local-only; downloads happen only through scripts/download_models.py.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Ensure symlinks are disabled on Windows to prevent permission errors
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logger = logging.getLogger(__name__)

CACHE_BUCKETS = ("embeddings", "rag", "coder", "sd")

SD15_REPO_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SD15_REMOTE_FILENAME = "v1-5-pruned-emaonly.safetensors"
SD15_LOCAL_FILENAME = "sd15.safetensors"

MODEL_ALIASES = {
    "sd1.5": SD15_REPO_ID,
    "stable-diffusion-1.5": SD15_REPO_ID,
    SD15_REMOTE_FILENAME: SD15_REPO_ID,
    SD15_LOCAL_FILENAME: SD15_REPO_ID,
}


@dataclass(frozen=True)
class RequiredModel:
    role: str
    repo_id: str


def normalize_model_id(model_id: str) -> str:
    clean = str(model_id or "").strip()
    return MODEL_ALIASES.get(clean, clean)


def hf_cache_folder_name(repo_id: str) -> str:
    return f"models--{normalize_model_id(repo_id).replace('/', '--')}"


def resolve_cache_dir(project_root: Path, cache_dir: str | Path) -> Path:
    raw = Path(str(cache_dir))
    if raw.is_absolute():
        return raw
    return project_root / raw


def create_cache_layout(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for bucket in CACHE_BUCKETS:
        (cache_dir / bucket).mkdir(parents=True, exist_ok=True)


def required_models_from_config(cfg: dict) -> list[RequiredModel]:
    return [
        RequiredModel("embeddings", str(cfg.get("embedding_id", "sentence-transformers/all-MiniLM-L6-v2"))),
        RequiredModel("rag", str(cfg.get("model_id", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))),
        RequiredModel("coder", str(cfg.get("coder_model_id", "Qwen/Qwen2.5-Coder-0.5B-Instruct"))),
        RequiredModel("sd", normalize_model_id(str(cfg.get("image_model_id", "sd1.5")))),
    ]


def sd_single_file_path(cache_dir: Path) -> Path:
    return cache_dir / "sd" / SD15_LOCAL_FILENAME


def candidate_sd_files(cache_dir: Path, model_id: str = "sd1.5") -> list[Path]:
    exact = sd_single_file_path(cache_dir)
    return [exact] if exact.is_file() else []


def _snapshot_from_hf_root(hf_root: Path) -> Path | None:
    ref = hf_root / "refs" / "main"
    if ref.exists():
        snapshot = hf_root / "snapshots" / ref.read_text(encoding="utf-8").strip()
        if snapshot.exists():
            return snapshot
    snapshots_dir = hf_root / "snapshots"
    if snapshots_dir.exists():
        snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
        snapshots.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        if snapshots:
            return snapshots[0]
    return None


def candidate_model_roots(cache_dir: Path, repo_id: str, preferred_role: str | None = None) -> list[Path]:
    normalized = normalize_model_id(repo_id)
    candidates: list[Path] = []

    explicit = Path(normalized)
    if explicit.exists():
        candidates.append(explicit)

    roles: list[str] = []
    if preferred_role:
        roles.append(preferred_role)
    roles.extend(role for role in CACHE_BUCKETS if role not in roles)

    folder_name = hf_cache_folder_name(normalized)
    for role in roles:
        role_root = cache_dir / role
        direct = role_root / normalized
        if direct.exists():
            candidates.append(direct)
        hf_root = role_root / folder_name
        if hf_root.exists():
            snapshot = _snapshot_from_hf_root(hf_root)
            if snapshot is not None:
                candidates.append(snapshot)
            candidates.append(hf_root)

    legacy_hf_root = cache_dir / folder_name
    if legacy_hf_root.exists():
        snapshot = _snapshot_from_hf_root(legacy_hf_root)
        if snapshot is not None:
            candidates.append(snapshot)
        candidates.append(legacy_hf_root)

    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(candidate)
    return unique


def model_exists(cache_dir: Path, repo_id: str, role: str) -> bool:
    if role == "sd":
        return bool(candidate_sd_files(cache_dir, repo_id))
    return bool(candidate_model_roots(cache_dir, repo_id, preferred_role=role))


def migrate_legacy_cache(cache_dir: Path, required_models: Iterable[RequiredModel]) -> None:
    create_cache_layout(cache_dir)
    for item in required_models:
        if item.role == "sd":
            continue
        folder_name = hf_cache_folder_name(item.repo_id)
        legacy = cache_dir / folder_name
        target = cache_dir / item.role / folder_name
        if target.exists() or not legacy.exists():
            continue
        try:
            logger.info("[MODEL_CACHE] Moving legacy cache %s -> %s", legacy, target)
            shutil.move(str(legacy), str(target))
        except Exception:
            logger.exception("[MODEL_CACHE] Could not migrate %s; keeping legacy fallback active.", legacy)


def download_model(repo_id: str, cache_dir: Path, role: str) -> Path:
    if role == "sd":
        existing = candidate_sd_files(cache_dir, repo_id)
        if existing:
            return existing[0]
        return download_sd15_single_file(cache_dir)

    from huggingface_hub import snapshot_download

    normalized = normalize_model_id(repo_id)
    role_cache = cache_dir / role
    role_cache.mkdir(parents=True, exist_ok=True)
    logger.info("[MODEL_CACHE] Downloading %s into %s", normalized, role_cache)
    previous_env = {
        key: os.environ.get(key)
        for key in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_SYMLINKS")
    }
    try:
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        for key in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ.pop(key, None)
        
        # We use local_dir and local_dir_use_symlinks=False to bypass Windows permission issues
        target_dir = role_cache / normalized
        return Path(
            snapshot_download(
                repo_id=normalized,
                cache_dir=str(role_cache),
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
                local_files_only=False,
                resume_download=True,
            )
        )
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


SD15_MIN_FILE_SIZE_BYTES = 4_000_000_000  # valid file is ~4.27 GB


def download_sd15_single_file(cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    target = sd_single_file_path(cache_dir)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists AND is large enough (not corrupt/truncated)
    if target.is_file():
        file_size = target.stat().st_size
        if file_size >= SD15_MIN_FILE_SIZE_BYTES:
            return target
        # File exists but is too small — corrupt/incomplete download
        logger.warning(
            "[MODEL_CACHE] SD1.5 file is too small (%.2f GB, expected >= %.2f GB). "
            "Deleting corrupt file and re-downloading.",
            file_size / (1024 ** 3),
            SD15_MIN_FILE_SIZE_BYTES / (1024 ** 3),
        )
        try:
            target.unlink()
        except OSError as exc:
            logger.error("[MODEL_CACHE] Could not delete corrupt SD file: %s", exc)
    logger.info(
        "[MODEL_CACHE] Downloading SD1.5 single file %s/%s -> %s",
        SD15_REPO_ID,
        SD15_REMOTE_FILENAME,
        target,
    )
    previous_env = {
        key: os.environ.get(key)
        for key in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_SYMLINKS")
    }
    try:
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
        for key in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ.pop(key, None)
        downloaded = Path(
            hf_hub_download(
                repo_id=SD15_REPO_ID,
                filename=SD15_REMOTE_FILENAME,
                cache_dir=str(target.parent / ".hf"),
                local_files_only=False,
                resume_download=True,
            )
        )
        shutil.copy2(downloaded, target)
        return target
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def resolve_required_model(cache_dir: Path, repo_id: str, role: str) -> Path | None:
    if role == "sd":
        files = candidate_sd_files(cache_dir, repo_id)
        return files[0] if files else None

    roots = candidate_model_roots(cache_dir, repo_id, preferred_role=role)
    if roots:
        return roots[0]
    return None


def prepare_model_cache(project_root: Path, cfg: dict) -> dict[str, str | list[str]]:
    cache_dir = resolve_cache_dir(project_root, cfg.get("cache_dir", "model_checkpoints"))
    create_cache_layout(cache_dir)
    required = required_models_from_config(cfg)
    migrate_legacy_cache(cache_dir, required)

    status: dict[str, str | list[str]] = {"cache_dir": str(cache_dir), "missing": []}
    missing: list[str] = []

    for item in required:
        model_path = resolve_required_model(cache_dir, item.repo_id, item.role)
        key = f"{item.role}_model_path"
        if model_path is None:
            missing.append(f"{item.role}:{normalize_model_id(item.repo_id)}")
            status[key] = ""
        else:
            status[key] = str(model_path)

    status["missing"] = missing
    manifest_path = cache_dir / "cache_manifest.json"
    try:
        manifest_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    except Exception:
        logger.exception("[MODEL_CACHE] Could not write cache manifest.")
    return status
