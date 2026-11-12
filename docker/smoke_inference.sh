#!/usr/bin/env bash
# End-to-end inference smoke test (requires NVIDIA GPU + running mono stack).
#
# Usage:
#   ./docker/smoke_inference.sh              # against current stack
#   TAG=latest ./docker/smoke_inference.sh --pull   # pull GHCR image, recreate api, then smoke
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
RUNNER_PY="$SCRIPT_DIR/smoke_inference_run.py"

TAG="${TAG:-latest}"
OWNER="${OWNER:-utahmobl}"
REGISTRY="${REGISTRY:-ghcr.io}"
TIMEOUT="${SMOKE_TIMEOUT:-1800}"
DO_PULL=0

for arg in "$@"; do
  case "$arg" in
    --pull) DO_PULL=1 ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

if [[ "$DO_PULL" == "1" ]]; then
  export OPENCAP_MONO_IMAGE="${OPENCAP_MONO_IMAGE:-${REGISTRY}/${OWNER}/opencap-mono:${TAG}}"
  echo "Pulling $OPENCAP_MONO_IMAGE ..."
  docker pull "$OPENCAP_MONO_IMAGE"
  docker tag "$OPENCAP_MONO_IMAGE" opencap-mono:latest
  cd "$SCRIPT_DIR"
  docker compose -f "$COMPOSE_FILE" up -d --force-recreate --no-build mono-api mono-worker
  echo "Waiting for mono-api healthy..."
  for _ in $(seq 1 72); do
    if docker inspect --format '{{.State.Health.Status}}' docker-mono-api-1 2>/dev/null | grep -qx healthy; then
      break
    fi
    sleep 5
  done
fi

if ! docker inspect docker-mono-worker-1 >/dev/null 2>&1; then
  echo "docker-mono-worker-1 not found — start the stack first." >&2
  exit 1
fi

# Ensure intrinsics present (entrypoint usually syncs; force if missing).
docker exec docker-mono-api-1 bash -lc '
  test -f camera_intrinsics/iPhone13,3/Deployed/cameraIntrinsics.pickle \
    || python installation/setup_camera_intrinsics.py
'

docker exec -e SMOKE_TIMEOUT="$TIMEOUT" docker-mono-worker-1 \
  python /workspace/opencap-mono/docker/smoke_inference_run.py
