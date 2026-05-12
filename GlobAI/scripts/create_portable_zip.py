"""
scripts/create_portable_zip.py
------------------------------
Create a minimal glob.zip containing only source code and config.
On a new PC, the user runs build.bat which auto-creates nexarag_env,
installs dependencies, and downloads models.
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_NAME = "glob.zip"
ZIP_PATH = PROJECT_ROOT / ZIP_NAME

# Directories to EXCLUDE entirely (case-insensitive match against relative parts)
EXCLUDE_DIRS = {
    "nexarag_env",
    "model_checkpoints",
    "__pycache__",
    "logs",
    "data",
    "runtime",
    ".git",
    ".hf",
    ".vscode",
    ".idea",
    ".gemini",
}

# File extensions to EXCLUDE
EXCLUDE_EXTENSIONS = {
    ".zip", ".exe", ".spec", ".log", ".bak", ".swp",
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".egg",
}

# Specific filenames to EXCLUDE
EXCLUDE_FILES = {
    "startup_debug.log",
    "glob.zip",
    "Thumbs.db",
    ".DS_Store",
    "# Game loop.py",  # junk file in coder/
}


def should_exclude(rel_path: Path) -> bool:
    """Return True if this path should be excluded from the zip."""
    # Check if any parent directory is in the exclude set
    for part in rel_path.parts:
        if part.lower() in {d.lower() for d in EXCLUDE_DIRS}:
            return True

    # Check file extension
    if rel_path.suffix.lower() in EXCLUDE_EXTENSIONS:
        return True

    # Check specific filenames
    if rel_path.name in EXCLUDE_FILES:
        return True

    return False


def create_zip() -> Path:
    """Create the portable zip file."""
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
        print(f"[ZIP] Removed existing {ZIP_NAME}")

    included: list[str] = []
    excluded: list[str] = []

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            root_path = Path(root)
            rel_root = root_path.relative_to(PROJECT_ROOT)

            # Prune excluded directories so os.walk doesn't descend into them
            dirs[:] = [
                d for d in dirs
                if d.lower() not in {x.lower() for x in EXCLUDE_DIRS}
            ]

            for filename in sorted(files):
                rel_file = rel_root / filename
                if should_exclude(rel_file):
                    excluded.append(str(rel_file))
                    continue

                full_path = root_path / filename
                # Use forward-slash paths inside zip for cross-platform compat
                arc_name = str(rel_file).replace("\\", "/")
                zf.write(full_path, arc_name)
                included.append(arc_name)

    size_kb = ZIP_PATH.stat().st_size / 1024
    size_mb = size_kb / 1024

    print(f"\n[ZIP] Created: {ZIP_PATH}")
    print(f"[ZIP] Size: {size_kb:.1f} KB ({size_mb:.2f} MB)")
    print(f"[ZIP] Files included: {len(included)}")
    print(f"[ZIP] Files excluded: {len(excluded)}")

    print("\n[ZIP] Contents:")
    for name in included:
        print(f"  + {name}")

    if excluded:
        print(f"\n[ZIP] Excluded {len(excluded)} file(s) (showing first 20):")
        for name in excluded[:20]:
            print(f"  - {name}")
        if len(excluded) > 20:
            print(f"  ... and {len(excluded) - 20} more")

    return ZIP_PATH


def main() -> int:
    print("=" * 50)
    print("  GlobAI Portable Zip Creator")
    print("=" * 50)

    zip_path = create_zip()

    print(f"\n[DONE] Portable archive ready: {zip_path}")
    print("[DONE] On a new PC:")
    print("       1. Extract glob.zip")
    print("       2. Run build.bat (creates env, installs deps, downloads models)")
    print("       3. Run run.bat to launch GlobAI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
