#!/usr/bin/env python3
"""Called by docker/smoke_inference.sh inside the worker container."""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path


def main() -> int:
    key = ""
    for line in Path("/workspace/opencap-mono/.env").read_text().splitlines():
        if line.startswith("MONO_API_KEY="):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
            break
    if not key:
        print("MONO_API_KEY missing in .env", file=sys.stderr)
        return 1

    paths = {
        "video_path": "/workspace/opencap-mono/examples/walking4/walking4_trimmed.mp4",
        "metadata_path": "/workspace/opencap-mono/ci/fixtures/smoke_metadata.yaml",
        "calib_path": "/workspace/opencap-mono/examples/walking4/calib.txt",
        "intrinsics_path": "/workspace/opencap-mono/camera_intrinsics/iPhone13,3/Deployed/cameraIntrinsics.pickle",
    }
    for p in paths.values():
        if not Path(p).exists():
            print(f"missing fixture: {p}", file=sys.stderr)
            return 1

    payload = {
        **paths,
        "estimate_local_only": True,
        "rerun": True,
        "session_id": "ci-smoke-inference",
        "activity": "walking",
    }
    req = urllib.request.Request(
        "http://mono-api:8000/run_mono",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-API-Key": key},
        method="POST",
    )
    print("POST http://mono-api:8000/run_mono ...", flush=True)
    t0 = time.time()
    with urllib.request.urlopen(
        req, timeout=int(os.environ.get("SMOKE_TIMEOUT", "1800"))
    ) as resp:
        body = json.loads(resp.read().decode())

    summary = {
        "message": body.get("message"),
        "predicted_activity": body.get("predicted_activity"),
        "trc_file_path": body.get("trc_file_path"),
        "scaled_model_file_path": body.get("scaled_model_file_path"),
        "ik_file_path": body.get("ik_file_path"),
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(json.dumps(summary, indent=2), flush=True)

    ok = (
        "success" in (body.get("message") or "").lower()
        and body.get("trc_file_path")
        and body.get("scaled_model_file_path")
        and body.get("ik_file_path")
    )
    print("SMOKE OK" if ok else "SMOKE FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
