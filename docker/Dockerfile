# GPU-enabled image — requires NVIDIA driver ≥ 520 (CUDA 11.8+) on the host.
# PyTorch 2.1.2+cu118 runs on CUDA 11.8 and all 12.x drivers.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH
ENV CONDA_ENV=opencap-mono
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV PYTHONUNBUFFERED=1
# Allow CUDA runtime to be visible inside the container
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip \
    gcc \
    g++ \
    git \
    python3-pip \
    python3-dev \
    python3-opencv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ncurses-base \
    libncurses5 \
    libncurses6 \
    libgtk-3-0 \
    libxrender1 \
    libgl1-mesa-dev \
    ffmpeg \
    libglu1-mesa \
    freeglut3 \
    mesa-common-dev \
    libglvnd0 \
    libglx0 \
    libopengl0 \
    && ldconfig \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda and create conda env
RUN wget --quiet --tries=3 https://repo.anaconda.com/miniconda/Miniconda3-py39_23.11.0-2-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh && \
    /opt/conda/bin/conda create -n $CONDA_ENV python=3.9 -y && \
    /opt/conda/bin/conda run -n $CONDA_ENV pip install --upgrade pip && \
    /opt/conda/bin/conda clean -afy

# Install PyTorch 2.1.2 + CUDA 11.8 (works on CUDA 11.8 and 12.x hosts)
RUN /opt/conda/bin/conda run -n $CONDA_ENV pip install \
    torch==2.1.2+cu118 \
    torchvision==0.16.2+cu118 \
    torchaudio==2.1.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118 \
    && rm -rf ~/.cache/pip

# Install OpenSim 4.4 first — its py39np121 build pins numpy to 1.21.x
RUN /opt/conda/bin/conda run -n $CONDA_ENV conda install -y -c opensim-org opensim=4.4=py39np121 && \
    /opt/conda/bin/conda clean -afy

# Install mmcv-full 1.7.2 via openmim (prebuilt CUDA wheel for pt2.1+cu118)
RUN /opt/conda/bin/conda run -n $CONDA_ENV pip install openmim && \
    /opt/conda/bin/conda run -n $CONDA_ENV mim install "mmcv-full==1.7.2" && \
    rm -rf ~/.cache/pip

# Copy and install pip requirements
WORKDIR /workspace/opencap-mono
COPY installation/requirements_slim.txt /workspace/opencap-mono/installation/requirements_slim.txt

# chumpy 0.70 needs setuptools (for pkg_resources) present at build time.
# Install it separately first to avoid isolated-build failures.
RUN /opt/conda/bin/conda run -n $CONDA_ENV pip install "setuptools>=60" "wheel" && \
    /opt/conda/bin/conda run -n $CONDA_ENV pip install "chumpy==0.70" --no-build-isolation && \
    rm -rf ~/.cache/pip

# Install remaining pip requirements.
# Pins numpy==1.21.6 and pandas==2.0.3 last — chumpy uses np.bool removed in numpy 1.24+.
RUN /opt/conda/bin/conda run -n $CONDA_ENV pip install -r installation/requirements_slim.txt && \
    /opt/conda/bin/conda run -n $CONDA_ENV pip install "numpy==1.21.6" "pandas==2.0.3" && \
    rm -rf ~/.cache/pip

# Install ViTPose (build in-place from submodule)
COPY WHAM/third-party/ViTPose /workspace/opencap-mono/WHAM/third-party/ViTPose

RUN /opt/conda/bin/conda run -n $CONDA_ENV pip install -v -e WHAM/third-party/ViTPose && \
    rm -rf ~/.cache/pip

# Install SLAHMR utilities (SMPL body model helpers — not SLAM)
COPY slahmr /workspace/opencap-mono/slahmr

RUN /opt/conda/bin/conda run -n $CONDA_ENV pip install -e slahmr/ && \
    rm -rf ~/.cache/pip

# Copy the rest of the codebase
COPY . /workspace/opencap-mono

ENV PATH=/opt/conda/envs/opencap-mono/bin:$PATH
ENV PYTHONPATH=/workspace/opencap-mono

EXPOSE 8000 8501

RUN cat > /entrypoint.sh << 'EOF'
#!/bin/bash
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES"

if [ "$1" = "api" ]; then
    exec /opt/conda/envs/opencap-mono/bin/uvicorn mono_api:app --host 0.0.0.0 --port 8000
elif [ "$1" = "api-dev" ]; then
    exec /opt/conda/envs/opencap-mono/bin/uvicorn mono_api:app --host 0.0.0.0 --port 8000 --reload
elif [ "$1" = "streamlit" ]; then
    exec /opt/conda/envs/opencap-mono/bin/streamlit run validation/app.py --server.port 8501 --server.address 0.0.0.0
elif [ "$1" = "worker" ]; then
    exec /opt/conda/envs/opencap-mono/bin/python opencap/app.py
else
    echo "Usage: docker run <image> [api|api-dev|streamlit|worker]"
    exit 1
fi
EOF
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["api"]
