#!/usr/bin/env python3
"""
Download / update CameraIntrinsics from opencap-org/opencap-core.

By default, downloads if missing and re-syncs when the remote
CameraIntrinsics folder has newer commits than the last sync.

Usage:
  python setup_camera_intrinsics.py              # sync if missing or outdated
  python setup_camera_intrinsics.py --update     # force re-download
  python setup_camera_intrinsics.py --check      # print status; exit 1 if outdated/missing
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_URL = "https://github.com/opencap-org/opencap-core.git"
REPO_API = "https://api.github.com/repos/opencap-org/opencap-core"
SPARSE_PATH = "CameraIntrinsics"
SOURCE_COMMIT_FILE = ".source_commit"
USER_AGENT = "opencap-mono-intrinsics-sync"


def run_command(cmd: list[str], cwd=None):
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=cwd)
    return result.stdout.strip()


def _http_get_json(url: str) -> object:
    req = urllib.request.Request(
        url, headers={"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def fetch_remote_intrinsics_commit() -> str:
    """Latest commit SHA that touched CameraIntrinsics on main."""
    url = f"{REPO_API}/commits?path={SPARSE_PATH}&sha=main&per_page=1"
    data = _http_get_json(url)
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"No commits found for {SPARSE_PATH} on main")
    return data[0]["sha"]


def read_local_commit(camera_intrinsics_dir: Path) -> str | None:
    marker = camera_intrinsics_dir / SOURCE_COMMIT_FILE
    if not marker.is_file():
        return None
    return marker.read_text().strip() or None


def write_local_commit(camera_intrinsics_dir: Path, sha: str) -> None:
    (camera_intrinsics_dir / SOURCE_COMMIT_FILE).write_text(sha + "\n")


def local_has_devices(camera_intrinsics_dir: Path) -> bool:
    if not camera_intrinsics_dir.is_dir():
        return False
    return any(
        d.is_dir() and any(p in d.name for p in ("iPhone", "iPad", "iPod"))
        for d in camera_intrinsics_dir.iterdir()
    )


def sync_status(project_root: Path) -> tuple[str, str | None, str | None]:
    """
    Returns (status, local_sha, remote_sha).
    status is one of: missing, current, outdated, unknown
    """
    camera_intrinsics_dir = project_root / "camera_intrinsics"
    local_sha = (
        read_local_commit(camera_intrinsics_dir)
        if local_has_devices(camera_intrinsics_dir)
        else None
    )
    if not local_has_devices(camera_intrinsics_dir):
        try:
            remote_sha = fetch_remote_intrinsics_commit()
        except Exception:
            remote_sha = None
        return "missing", None, remote_sha

    try:
        remote_sha = fetch_remote_intrinsics_commit()
    except Exception as exc:
        print(
            f"Warning: could not check remote CameraIntrinsics: {exc}", file=sys.stderr
        )
        return "unknown", local_sha, None

    if local_sha is None:
        # Present but never stamped (legacy install) — treat as outdated so we refresh.
        return "outdated", None, remote_sha
    if local_sha == remote_sha:
        return "current", local_sha, remote_sha
    return "outdated", local_sha, remote_sha


def download_intrinsics(project_root: Path, remote_sha: str | None = None) -> None:
    camera_intrinsics_dir = project_root / "camera_intrinsics"
    temp_dir = project_root / "temp_opencap_core"
    staging_dir = project_root / ".camera_intrinsics.staging"
    backup_dir = project_root / ".camera_intrinsics.backup"

    # Recover the last good copy if a previous process was interrupted mid-swap.
    if backup_dir.exists() and not camera_intrinsics_dir.exists():
        backup_dir.rename(camera_intrinsics_dir)

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    try:
        print(f"Cloning {REPO_URL} (sparse, depth=1)...")
        run_command(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                REPO_URL,
                str(temp_dir),
            ],
            cwd=project_root,
        )
        run_command(["git", "sparse-checkout", "set", SPARSE_PATH], cwd=temp_dir)

        source_dir = temp_dir / SPARSE_PATH
        print(f"Staging {SPARSE_PATH} for an atomic update...")
        shutil.copytree(source_dir, staging_dir)

        # Prefer the path-specific commit from the API; fall back to clone HEAD.
        sha = remote_sha or run_command(["git", "rev-parse", "HEAD"], cwd=temp_dir)
        write_local_commit(staging_dir, sha)

        # Keep the installed directory intact until the replacement is complete.
        if camera_intrinsics_dir.exists():
            camera_intrinsics_dir.rename(backup_dir)
        try:
            staging_dir.rename(camera_intrinsics_dir)
        except Exception:
            if backup_dir.exists() and not camera_intrinsics_dir.exists():
                backup_dir.rename(camera_intrinsics_dir)
            raise
        if backup_dir.exists():
            shutil.rmtree(backup_dir)

        models = sorted(
            d.name
            for d in camera_intrinsics_dir.iterdir()
            if d.is_dir() and any(p in d.name for p in ("iPhone", "iPad", "iPod"))
        )
        print(f"Done. {len(models)} device models available (source {sha[:12]}).")
        print("Latest models:", ", ".join(models[-5:]))

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def ensure_intrinsics(project_root: Path, force: bool = False) -> bool:
    """
    Ensure camera_intrinsics/ matches remote. Returns True if a download ran.
    """
    if force:
        try:
            remote_sha = fetch_remote_intrinsics_commit()
        except Exception as exc:
            print(
                f"Warning: could not resolve remote commit ({exc}); cloning tip of main."
            )
            remote_sha = None
        download_intrinsics(project_root, remote_sha=remote_sha)
        return True

    status, local_sha, remote_sha = sync_status(project_root)
    if status == "current":
        print(f"camera_intrinsics/ is up to date ({local_sha[:12]}).")
        return False
    if status == "unknown":
        print("camera_intrinsics/ present; remote check failed — leaving as-is.")
        return False
    if status == "missing":
        print("camera_intrinsics/ missing — downloading...")
    else:
        local_disp = local_sha[:12] if local_sha else "unstamped"
        remote_disp = remote_sha[:12] if remote_sha else "unknown"
        print(
            f"camera_intrinsics/ outdated (local {local_disp} → remote {remote_disp}) — updating..."
        )

    download_intrinsics(project_root, remote_sha=remote_sha)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download/update camera intrinsics from opencap-org/opencap-core."
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Force re-download even if camera_intrinsics/ matches remote.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check status; exit 0 if current, 1 if missing/outdated.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.absolute()

    if args.check:
        status, local_sha, remote_sha = sync_status(project_root)
        print(f"status={status}")
        print(f"local={local_sha or ''}")
        print(f"remote={remote_sha or ''}")
        sys.exit(0 if status == "current" else 1)

    try:
        ensure_intrinsics(project_root, force=args.update)
    except urllib.error.URLError as exc:
        print(f"Network error syncing intrinsics: {exc}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or str(exc)).strip()
        print(f"git error syncing intrinsics: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
