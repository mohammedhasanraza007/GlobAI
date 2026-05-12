from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = PROJECT_ROOT.parent / "GlobAI_release"
OUTPUT_ZIP = RELEASE_ROOT / "GlobAI_GitHub_Source.zip"
TEMP_ZIP = RELEASE_ROOT / "GlobAI_GitHub_Source.zip.tmp"

EXCLUDED_DIRS = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    "venv",
    ".venv",
    "env",
    "ENV",
    "nexarag_env",
    "model_checkpoints",
    "build",
    "dist",
    "release",
    "logs",
    "downloads",
    ".ipynb_checkpoints",
}

EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".log",
    ".tmp",
    ".zip",
    ".exe",
    ".spec",
    ".bat",
    ".vbs",
}

EXCLUDED_FILES = {
    "debug_launch_stdout.txt",
    "debug_launch_stderr.txt",
    "startup_debug.log",
    "GlobAI.zip",
    "launcher.py",
    "check_imports.py",
    "verify_headless.py",
}


def should_exclude(path: Path) -> bool:
    relative = path.relative_to(PROJECT_ROOT)
    if any(part in EXCLUDED_DIRS for part in relative.parts):
        return True
    if relative.parts[:2] == ("data", "vector_db"):
        return True
    if relative.parts[:2] == ("data", "outputs"):
        return True
    if path.name in EXCLUDED_FILES:
        return True
    return path.suffix.lower() in EXCLUDED_SUFFIXES


def build_zip() -> Path:
    RELEASE_ROOT.mkdir(parents=True, exist_ok=True)
    if TEMP_ZIP.exists():
        TEMP_ZIP.unlink()

    with ZipFile(TEMP_ZIP, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(PROJECT_ROOT.rglob("*")):
            if path.is_dir() or should_exclude(path):
                continue
            archive.write(path, path.relative_to(PROJECT_ROOT))

    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
    TEMP_ZIP.replace(OUTPUT_ZIP)
    return OUTPUT_ZIP


if __name__ == "__main__":
    output = build_zip()
    print(output)
