#!/usr/bin/env python3
"""
Download / update CameraIntrinsics from opencap-org/opencap-core.

Usage:
  python setup_camera_intrinsics.py           # download if not present
  python setup_camera_intrinsics.py --update  # force re-download (replaces existing)
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/opencap-org/opencap-core.git"
SPARSE_PATH = "CameraIntrinsics"


def run_command(cmd, cwd=None):
    result = subprocess.run(
        cmd, shell=True, check=True, capture_output=True, text=True, cwd=cwd
    )
    return result.stdout.strip()


def download_intrinsics(project_root: Path, force: bool = False):
    camera_intrinsics_dir = project_root / "camera_intrinsics"
    temp_dir = project_root / "temp_opencap_core"

    if camera_intrinsics_dir.exists() and any(camera_intrinsics_dir.iterdir()):
        if not force:
            print("camera_intrinsics/ already exists. Use --update to re-download.")
            return
        print("Removing existing camera_intrinsics/ for fresh download...")
        shutil.rmtree(camera_intrinsics_dir)

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    try:
        print(f"Cloning {REPO_URL} (sparse, depth=1)...")
        run_command(
            f"git clone --depth 1 --filter=blob:none --sparse {REPO_URL} {temp_dir}",
            cwd=project_root,
        )
        run_command(f"git sparse-checkout set {SPARSE_PATH}", cwd=temp_dir)

        source_dir = temp_dir / SPARSE_PATH
        print(f"Copying {SPARSE_PATH} → camera_intrinsics/ ...")
        shutil.copytree(source_dir, camera_intrinsics_dir)

        models = sorted(
            d.name
            for d in camera_intrinsics_dir.iterdir()
            if d.is_dir() and any(p in d.name for p in ("iPhone", "iPad", "iPod"))
        )
        print(f"Done. {len(models)} device models available.")
        print("Latest models:", ", ".join(models[-5:]))

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(description="Download/update camera intrinsics.")
    parser.add_argument(
        "--update", action="store_true",
        help="Force re-download even if camera_intrinsics/ already exists."
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.absolute()
    download_intrinsics(project_root, force=args.update)


if __name__ == "__main__":
    main()
