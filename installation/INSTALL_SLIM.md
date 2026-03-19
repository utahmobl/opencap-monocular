# Bare-metal Installation (Slim)

Tested on Ubuntu 20.04/22.04, Python 3.9, CUDA 11.8 and 12.x (driver ≥ 520).

> **Prefer Docker?** See [docker/README.md](../docker/README.md) — it's the recommended deployment path for production machines.

## What's changed vs the original guide
- DPVO (visual odometry SLAM) removed — not needed for static cameras; WHAM falls back to zero camera rotation automatically.
- PyTorch3D removed — optional visualization dependency, not used in the pipeline.
- PyTorch updated: `1.11 + cu113` → `2.1.2 + cu118` (works on CUDA 11.8 and 12.x drivers).
- mmcv updated: `1.3.9` → `1.7.2` (installed via `openmim` for correct CUDA/PyTorch matching).

---

## 1. Clone

```bash
git clone https://github.com/utahmobl/opencap-monocular.git --recursive
cd opencap-monocular
```

## 2. Create conda environment

```bash
conda create -n opencap-mono-slim python=3.9 -y
conda activate opencap-mono-slim
```

## 3. Install PyTorch 2.1.2 (CUDA 11.8 build)

Works on any system with NVIDIA driver ≥ 520 (CUDA 11.8 through 12.x).

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

## 4. Install OpenSim 4.4

Install OpenSim via conda **before** the pip dependencies.

```bash
conda install -c opensim-org opensim=4.4=py39np121 -y
```

> **numpy constraint**: OpenSim 4.4 requires `numpy==1.21.6`. The `requirements_slim.txt` pins compatible versions for all downstream packages. Do not upgrade numpy beyond 1.23.

## 5. Install mmcv-full via openmim

`openmim` selects the prebuilt `mmcv-full` wheel that matches your exact PyTorch + CUDA version.

```bash
pip install openmim
mim install "mmcv-full==1.7.2"
```

## 6. Install pip dependencies

```bash
pip install -r installation/requirements_slim.txt
```

## 7. Install ViTPose

```bash
pip install -v -e WHAM/third-party/ViTPose
```

## 8. Set up SLAHMR utilities (SMPL body model)

```bash
pip install -e slahmr/
```

## 9. Download camera intrinsics

```bash
cd installation && make setup-intrinsics && cd ..
```

To update intrinsics to the latest device models:
```bash
cd installation && make update-intrinsics && cd ..
```

## 10. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your API_TOKEN, MONO_API_KEY, etc.
```

## 11. Verify

```bash
python -c "
import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
import mmcv; print('mmcv:', mmcv.__version__)
import opensim; print('opensim:', opensim.__version__)
import mmpose; print('ViTPose/mmpose:', mmpose.__version__)
import optimization; print('optimization: OK')
print('Installation successful.')
"
```

---

## Optional: VideoLLaMA3 activity classifier

VideoLLaMA3 automatically classifies the activity in each video (walking, squatting, sit-to-stand, etc.) to select the correct optimization parameters. Without it the pipeline falls back to ankle-velocity gait detection and a generic "other" category.

For bare-metal use, clone the repo and run it separately:

```bash
git clone https://github.com/DAMO-NLP-SG/VideoLLaMA3.git ../VideoLLaMA3
# Follow VideoLLaMA3's own README for dependencies
# Then set in .env:
# VIDEOLLAMA_URL=http://localhost:8400
```

In Docker, VideoLLaMA3 is included as a service — see [docker/README.md](../docker/README.md).

---

## Notes

- **DPVO is not installed** — WHAM detects its absence at runtime and uses a static-camera fallback (zero angular velocity). No code changes needed.
- **Models / weights**: Download pre-trained WHAM weights as described in `WHAM/README.md` if not already present.
- **Docker**: See [docker/README.md](../docker/README.md) for the containerized version of this setup.
