#!/bin/bash
# Quick smoke-test for NVIDIA Docker on this machine.
# Uses --runtime=nvidia + explicit UVM devices (--gpus all / CDI mode not available here).

set -e

echo "=== 1. Host NVIDIA driver ==="
nvidia-smi | head -10

echo -e "\n=== 2. Docker daemon runtime ==="
docker info | grep -E "Runtimes|Default Runtime"

echo -e "\n=== 3. nvidia-smi inside container ==="
docker run --rm --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --device /dev/nvidia-uvm \
  --device /dev/nvidia-uvm-tools \
  nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi | head -10

echo -e "\n=== 4. PyTorch CUDA inside opencap-mono image ==="
docker run --rm --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --device /dev/nvidia-uvm \
  --device /dev/nvidia-uvm-tools \
  --entrypoint /opt/conda/envs/opencap-mono/bin/python \
  opencap-mono:latest \
  -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo -e "\nAll checks passed."
