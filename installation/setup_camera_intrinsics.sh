#!/bin/bash

# Setup / sync CameraIntrinsics from opencap-core.
# Delegates to setup_camera_intrinsics.py (auto-updates when remote changes).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
else
    PYTHON=python
fi

exec "$PYTHON" "$SCRIPT_DIR/setup_camera_intrinsics.py" "$@"
