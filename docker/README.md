# Docker — opencap-mono

## Files in this directory


| File                    | Purpose                                                       |
| ----------------------- | ------------------------------------------------------------- |
| `Dockerfile`            | Main GPU image — API, worker, streamlit (all share one image) |
| `Dockerfile.logs`       | Lightweight image — log viewer and Slack monitor              |
| `docker-compose.yml`    | All services                                                  |
| `pull-images.sh`        | Pull GHCR images and recreate the stack                       |
| `test-nvidia-docker.sh` | Quick sanity-check for NVIDIA Docker                          |


The `.dockerignore` that controls the build context lives at the **repo root** (not here).

---

## Prerequisites


| Requirement                | Notes                                            |
| -------------------------- | ------------------------------------------------ |
| NVIDIA driver ≥ 520        | `nvidia-smi` must work on the host               |
| `nvidia-container-toolkit` | Installed and `nvidia-container-runtime` on PATH |
| `docker buildx`            | See install snippet below                        |
| `.env` file at repo root   | `cp .env.example .env` then fill in your keys    |


Install `docker buildx` without root:

```bash
mkdir -p ~/.docker/cli-plugins
curl -sL "https://github.com/docker/buildx/releases/download/v0.19.3/buildx-v0.19.3.linux-amd64" \
  -o ~/.docker/cli-plugins/docker-buildx && chmod +x ~/.docker/cli-plugins/docker-buildx
```

Verify: `docker buildx version`

> **Snap curl:** If you use the Snap version of `curl`, it may not write to `~/.docker` (hidden dir). Use the native package: `sudo apt install curl`, then re-run the install snippet above.

---

## One-time build

Three images need to be built. All are fast after the first run thanks to BuildKit layer caching.

**Option A — pull from GitHub Actions / GHCR (preferred on new machines):**

Images are published by GitHub Actions to:

- `ghcr.io/utahmobl/opencap-mono` / `opencap-mono-logs` (this repo)
- `ghcr.io/utahmobl/videollama3-api` ([VideoLLaMA3_api](https://github.com/utahmobl/VideoLLaMA3_api))

```bash
# PAT needs read:packages (authorize SSO for utahmobl if prompted)
echo "$GHCR_TOKEN" | docker login ghcr.io -u YOUR_GITHUB_USER --password-stdin

# Pull + retag + recreate stack (production machines should use TAG=latest from main)
TAG=latest ./docker/pull-images.sh
TAG=latest ./docker/pull-images.sh --with-videollama   # also pull/restart VideoLLaMA
# Staging / pre-prod from the dev branch image:
# TAG=dev ./docker/pull-images.sh --with-videollama
# After pull, verify inference:
# ./docker/smoke_inference.sh
```

Or manually:

```bash
export OPENCAP_MONO_IMAGE=ghcr.io/utahmobl/opencap-mono:latest
export OPENCAP_LOGS_IMAGE=ghcr.io/utahmobl/opencap-mono-logs:latest
export VIDEOLLAMA_IMAGE=ghcr.io/utahmobl/videollama3-api:latest
docker compose -f docker/docker-compose.yml pull
docker compose -f docker/docker-compose.yml up -d
```

`latest` is published from each repo’s default branch; branch builds also get a tag named after the branch (e.g. `mobl_mono_1`).

> The repo is still bind-mounted at runtime for code, `WHAM/checkpoints/`, and `Data/`. Pulling the image replaces the heavy conda/CUDA layer build — you still need a checkout of this repo on the machine.

**Option B — build locally:**

```bash
# Main image — API, worker, streamlit (~5-10 min first time)
docker buildx build -f docker/Dockerfile -t opencap-mono:latest .

# Lightweight logs/monitor image (~15 s)
docker buildx build -f docker/Dockerfile.logs -t opencap-mono-logs:latest .

# VideoLLaMA3 activity classifier (~5-10 min first time; model downloads at first startup)
docker buildx build -f ../VideoLLaMA3_api/Dockerfile -t videollama3-api:latest ../VideoLLaMA3_api
```

> **Note:** The VideoLLaMA3 build context is the sibling `../VideoLLaMA3_api` directory
> ([github.com/utahmobl/VideoLLaMA3_api](https://github.com/utahmobl/VideoLLaMA3_api)).
> The model (`DAMO-NLP-SG/VideoLLaMA3-2B`, ~3 GB) is downloaded from HuggingFace on
> first container start and cached in `~/.cache/huggingface` on the host.

---

## Start / stop

**Full production stack** (API + worker + VideoLLaMA3 + Slack monitor + log viewer):

```bash
docker compose -f docker/docker-compose.yml up mono-api mono-worker videollama mono-slack mono-logs -d
```

**Minimal** (API + worker only):

```bash
docker compose -f docker/docker-compose.yml up mono-api mono-worker -d
```

**Stop everything:**

```bash
docker compose -f docker/docker-compose.yml down
```

**Restart a single service:**

```bash
docker compose -f docker/docker-compose.yml restart mono-api
```

---

## Logs

Follow live logs for one or more services:

```bash
docker compose -f docker/docker-compose.yml logs -f mono-api
docker compose -f docker/docker-compose.yml logs -f mono-worker
docker compose -f docker/docker-compose.yml logs -f mono-api mono-worker
```

Or open the **web log viewer** at **[http://localhost:8888](http://localhost:8888)** (requires `mono-logs` running).

---

## Rebuild after changes

The repo root is volume-mounted into every container, so **Python code edits are live** — no rebuild needed.

Rebuild only when you change:

- `docker/Dockerfile` or `docker/Dockerfile.logs`
- `installation/requirements_slim.txt`
- `WHAM/third-party/ViTPose/` or `slahmr/` (installed packages)

```bash
docker buildx build -f docker/Dockerfile -t opencap-mono:latest . \
  && docker compose -f docker/docker-compose.yml up -d --force-recreate mono-api mono-worker
```

---

## Run a one-off command

```bash
# Interactive shell
docker run --rm -it --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
  -v $(pwd):/workspace/opencap-mono \
  --entrypoint bash \
  opencap-mono:latest

# Single Python command
docker run --rm --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
  -v $(pwd):/workspace/opencap-mono \
  --entrypoint /opt/conda/envs/opencap-mono/bin/python \
  opencap-mono:latest -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Services reference


| Service          | Image               | Port                    | Description                                                                         |
| ---------------- | ------------------- | ----------------------- | ----------------------------------------------------------------------------------- |
| `mono-api`       | `opencap-mono`      | none (internal only)    | Production API — only the local worker reaches it                                   |
| `mono-worker`    | `opencap-mono`      | none                    | Polls opencap server, calls `mono-api`                                              |
| `mono-streamlit` | `opencap-mono`      | `127.0.0.1:8503 → 8501` | Validation / visualisation UI                                                       |
| `videollama`     | `videollama3-api`   | `127.0.0.1:8400 → 8400` | VideoLLaMA3 activity classifier — called by `mono-api` for activity detection       |
| `mono-logs`      | `opencap-mono-logs` | `127.0.0.1:8888 → 8888` | Web log viewer (reads container logs via Docker socket)                             |
| `mono-slack`     | `opencap-mono-logs` | none                    | Background monitor — sends Slack alerts on service changes, errors, resource spikes |


`**mono-api` is intentionally not exposed** to the host network. Only `mono-worker` on the same machine reaches it via Docker's internal DNS (`http://mono-api:8000/`). Each machine on the network runs its own independent stack with no port conflicts.

---

## Slack notifications

Slack works at two levels:

1. **Inline** — `mono-api` and `mono-worker` send Slack messages for individual trial completions and errors directly (reads `SLACK_WEBHOOK_URL` from `.env`).
2. **Monitor** — `mono-slack` polls every 60 s, sending alerts for container status changes, error spikes, and resource thresholds (CPU/memory/disk > 90 %).

Both are automatically enabled when `SLACK_WEBHOOK_URL` is set in `.env`.

---

## Environment variables (`.env`)

```
API_TOKEN=<opencap api token>
API_URL=https://dev.opencap.ai/
MONO_API_URL=http://127.0.0.1:8000/   # bare-metal default; Docker overrides this automatically
MONO_API_KEY=<your key>
REQUIRE_API_KEY=true
SLACK_WEBHOOK_URL=<optional>
```

---

## Troubleshooting

**CUDA not available inside container**

`--gpus all` (CDI mode) does not work on this system. Always use:

```bash
--runtime=nvidia --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools
```

The `docker-compose.yml` already does this. Run `./docker/test-nvidia-docker.sh` to verify.

**Port already in use**

`mono-logs` uses `8888`, `mono-streamlit` uses `8503`. Change the host-side port in `docker-compose.yml` if needed:

```yaml
ports:
  - "8899:8888"
```

**Container exits immediately**

```bash
docker compose -f docker/docker-compose.yml logs mono-api
```

Common causes: missing `.env`, wrong `API_TOKEN`, missing model checkpoints under `WHAM/checkpoints/`.

**Stale image after `git pull`**

```bash
docker buildx build -f docker/Dockerfile -t opencap-mono:latest . \
  && docker compose -f docker/docker-compose.yml up -d --force-recreate mono-api mono-worker mono-slack mono-logs
```

