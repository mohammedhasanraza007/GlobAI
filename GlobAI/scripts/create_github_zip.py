"""
create_github_zip.py
--------------------
Creates a lightweight, source-only GitHub-ready ZIP of GlobAI.
Output: GlobAI_GitHub_Source.zip in the project root.

Excludes:
  - nexarag_env/  (virtual environment)
  - model_checkpoints/  (downloaded models)
  - logs/, build/, dist/, release/, __pycache__/
  - .exe, .spec, .pyc, .log, .tmp files
  - build_all.py, setup_gui.py, uninstall_gui.py  (obsolete EXE scripts)
  - data/  (user data / vector databases)
  - outputs/, caches/, temp/
"""

import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ZIP = PROJECT_ROOT / "GlobAI_GitHub_Source.zip"

EXCLUDED_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    "release",
    "logs",
    "model_checkpoints",
    "nexarag_env",
    "venv",
    ".venv",
    "env",
    "ENV",
    "downloads",
    ".ipynb_checkpoints",
    "data",
    "outputs",
    "caches",
    "temp",
    "Library",   # E:\Library system DLLs, not part of source
    "bin",
    "share",
    "scratch",
    "GlobAI_release",
}

EXCLUDED_FILES = {
    # Obsolete EXE/PyInstaller scripts
    "build_all.py",
    "setup_gui.py",
    "uninstall_gui.py",
    "launcher.py",
    "launcher.vbs",
    "installer.bat",
    "verify_headless.py",
    "check_imports.py",
    # Logs and debug artifacts
    "startup_debug.log",
    "debug_launch_stdout.txt",
    "debug_launch_stderr.txt",
    # ZIP outputs
    "GlobAI_GitHub_Source.zip",
    "GlobAI.zip",
}

EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".log",
    ".tmp",
    ".exe",
    ".spec",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".pth",   # model weight files
    ".bin",   # model binary files
    ".safetensors",
}


def should_include(path: Path) -> bool:
    relative = path.relative_to(PROJECT_ROOT)

    # Exclude if any parent directory is excluded
    if any(part in EXCLUDED_DIRS for part in relative.parts):
        return False

    # Exclude specific filenames
    if path.name in EXCLUDED_FILES:
        return False

    # Exclude by suffix
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False

    return True


def create_zip():
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
        print(f"[INFO] Removed existing {OUTPUT_ZIP.name}")

    count = 0
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            root_path = Path(root)

            # Prune excluded dirs in-place so os.walk skips them
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

            for file in files:
                file_path = root_path / file
                if should_include(file_path):
                    rel_path = file_path.relative_to(PROJECT_ROOT)
                    zipf.write(file_path, rel_path)
                    count += 1
                    print(f"  [+] {rel_path}")

    size_mb = OUTPUT_ZIP.stat().st_size / 1_048_576
    print(f"\n[DONE] Created {OUTPUT_ZIP.name}")
    print(f"       {count} files | {size_mb:.2f} MB")
    print(f"       Path: {OUTPUT_ZIP}")


if __name__ == "__main__":
    create_zip()
