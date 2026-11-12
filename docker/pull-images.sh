#!/usr/bin/env bash
# Pull pre-built images from GitHub Container Registry and (re)start the stack.
#
# Prerequisites:
#   1. A GitHub PAT with read:packages (and SSO authorized for utahmobl if needed)
#   2. docker login ghcr.io -u YOUR_GITHUB_USER -p YOUR_PAT
#
# Usage:
#   ./docker/pull-images.sh              # pull mono/logs + recreate stack
#   ./docker/pull-images.sh --with-videollama  # also pull/restart VideoLLaMA
#   ./docker/pull-images.sh --no-up      # pull only
#   OWNER=utahmobl TAG=mobl_mono_1 ./docker/pull-images.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"

OWNER="${OWNER:-utahmobl}"
TAG="${TAG:-latest}"
VL_TAG="${VIDEOLLAMA_TAG:-latest}"
REGISTRY="${REGISTRY:-ghcr.io}"

WITH_VIDEOLLAMA=0
NO_UP=0
for arg in "$@"; do
  case "$arg" in
    --with-videollama) WITH_VIDEOLLAMA=1 ;;
    --no-up) NO_UP=1 ;;
    *)
      echo "Unknown arg: $arg" >&2
      exit 1
      ;;
  esac
done

export OPENCAP_MONO_IMAGE="${OPENCAP_MONO_IMAGE:-${REGISTRY}/${OWNER}/opencap-mono:${TAG}}"
export OPENCAP_LOGS_IMAGE="${OPENCAP_LOGS_IMAGE:-${REGISTRY}/${OWNER}/opencap-mono-logs:${TAG}}"
export VIDEOLLAMA_IMAGE="${VIDEOLLAMA_IMAGE:-${REGISTRY}/${OWNER}/videollama3-api:${VL_TAG}}"

echo "Pulling:"
echo "  $OPENCAP_MONO_IMAGE"
echo "  $OPENCAP_LOGS_IMAGE"
docker pull "$OPENCAP_MONO_IMAGE"
docker pull "$OPENCAP_LOGS_IMAGE"

docker tag "$OPENCAP_MONO_IMAGE" opencap-mono:latest
docker tag "$OPENCAP_LOGS_IMAGE" opencap-mono-logs:latest

if [[ "$WITH_VIDEOLLAMA" == "1" ]]; then
  echo "  $VIDEOLLAMA_IMAGE"
  docker pull "$VIDEOLLAMA_IMAGE"
  docker tag "$VIDEOLLAMA_IMAGE" videollama3-api:latest
  docker tag "$VIDEOLLAMA_IMAGE" videollama3:latest
fi

if [[ "$NO_UP" == "1" ]]; then
  echo "Pull complete (--no-up)."
  exit 0
fi

cd "$SCRIPT_DIR"
docker compose -f "$COMPOSE_FILE" up -d --force-recreate \
  mono-api mono-worker mono-streamlit mono-logs mono-slack

if [[ "$WITH_VIDEOLLAMA" == "1" ]]; then
  # Standalone container name used by mono-api / log-viewer by default.
  VL_PULL="$REPO_ROOT/../VideoLLaMA3_api/docker/pull-image.sh"
  if [[ -x "$VL_PULL" ]]; then
    "$VL_PULL"
  else
    docker compose -f "$COMPOSE_FILE" --profile videollama up -d --force-recreate videollama
  fi
fi

echo "Stack recreated from registry images."
echo "Repo checkout is still bind-mounted for code/checkpoints/Data."
