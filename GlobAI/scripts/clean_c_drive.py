import os
import shutil
from pathlib import Path

def nuke_globai_on_c():
    """Removes all known GlobAI artifacts from the C drive."""
    
    # 1. Primary installation directory (LOCALAPPDATA)
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        c_globai = Path(local_app_data) / "GlobAI"
        if c_globai.exists():
            print(f"Deleting {c_globai}...")
            shutil.rmtree(c_globai, ignore_errors=True)
            
    # 2. Roaming data (just in case)
    app_data = os.environ.get("APPDATA")
    if app_data:
        c_globai_roaming = Path(app_data) / "GlobAI"
        if c_globai_roaming.exists():
            print(f"Deleting {c_globai_roaming}...")
            shutil.rmtree(c_globai_roaming, ignore_errors=True)
            
    # 3. Temp files
    temp_dir = os.environ.get("TEMP")
    if temp_dir:
        temp_globai = Path(temp_dir) / "GlobAI"
        if temp_globai.exists():
            print(f"Deleting {temp_globai}...")
            shutil.rmtree(temp_globai, ignore_errors=True)
            
        # Specific cleanup script remnants
        cleanup_ps1 = Path(temp_dir) / "GlobAI_uninstall_cleanup.ps1"
        if cleanup_ps1.exists():
            print(f"Deleting {cleanup_ps1}...")
            cleanup_ps1.unlink()

    # 4. Desktop shortcuts
    # We'll check the standard locations
    user_profile = os.environ.get("USERPROFILE")
    if user_profile:
        desktop = Path(user_profile) / "Desktop"
        shortcut = desktop / "GlobAI.lnk"
        if shortcut.exists():
            print(f"Deleting shortcut {shortcut}...")
            shortcut.unlink()

    print("\nCleanup of C: drive complete.")
    print("All GlobAI project files have been removed from the primary drive.")

if __name__ == "__main__":
    nuke_globai_on_c()
