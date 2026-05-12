import zipfile
import os
from pathlib import Path

def create_git_release_zip():
    # Paths
    d_drive_release = Path("D:/GlobAI_release")
    e_drive_release = Path("E:/GlobAI_release")
    
    release_root = e_drive_release if e_drive_release.exists() else d_drive_release
    
    if not release_root.exists():
        print(f"Error: Release folder not found at {release_root}")
        return

    zip_path = release_root / "GlobAI_Setup_Git.zip"
    
    important_files = ["Setup.exe", "INSTALL.txt"]
    
    print(f"Creating zip at {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in important_files:
            file_path = release_root / file
            if file_path.exists():
                print(f"Adding {file}...")
                zipf.write(file_path, arcname=file)
            else:
                print(f"Warning: {file} not found in {release_root}")
                
    print(f"Done! Zip created at {zip_path}")

if __name__ == "__main__":
    create_git_release_zip()
