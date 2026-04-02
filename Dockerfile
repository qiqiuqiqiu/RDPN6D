FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
ENV CUDA_HOME=/usr/local/cuda

WORKDIR /workspace/RDPN6D

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    git \
    build-essential \
    ninja-build \
    pkg-config \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libjpeg-dev \
    zlib1g-dev \
    libopenexr-dev \
    openexr \
    libglfw3-dev \
    libglfw3 \
    libassimp-dev \
    libegl1 \
    libgl1-mesa-glx \
    libglib2.0-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libusb-1.0-0 \
    libusb-1.0-0-dev \
    udev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    python -m pip install --upgrade pip setuptools wheel

# PyTorch stack
RUN pip install \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Core framework dependencies that are easier to control separately
RUN pip install \
    mmcv==1.7.2 \
    pytorch-lightning==1.9.5 \
    tensorboardX==2.5.1 \
    fvcore==0.1.5.post20221221 \
    iopath==0.1.9 \
    pycocotools==2.0.8 \
    pypng==0.20220715.0 \
    transforms3d==0.4.2 \
    open3d==0.18.0 \
    opencv-python==4.5.5.62 \
    numpy==1.23.4 \
    pyrealsense2

# Detectron2 from source to match the repository expectation
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install the remaining Python dependencies from the repo.
# We filter out packages we already install explicitly above to reduce conflicts.
COPY requirements.txt /tmp/requirements.txt
RUN python - <<'PY'
from pathlib import Path
src = Path("/tmp/requirements.txt")
dst = Path("/tmp/requirements.docker.txt")
skip_prefixes = (
    "torch==",
    "torchvision==",
    "torchaudio==",
    "mmcv==",
    "mmcv-full==",
    "opencv-python==",
    "open3d==",
    "numpy==",
    "pycocotools==",
    "pypng==",
    "transforms3d==",
    "tensorboardX==",
    "fvcore==",
    "iopath==",
)
skip_exact = {
    "nvidia-cublas-cu11==11.11.3.6",
    "nvidia-cuda-cupti-cu11==11.8.87",
    "nvidia-cuda-nvrtc-cu11==11.8.89",
    "nvidia-cuda-runtime-cu11==11.8.89",
    "nvidia-cudnn-cu11==8.7.0.84",
    "nvidia-cufft-cu11==10.9.0.58",
    "nvidia-curand-cu11==10.3.0.86",
    "nvidia-cusolver-cu11==11.4.1.48",
    "nvidia-cusparse-cu11==11.7.5.86",
    "nvidia-nccl-cu11==2.19.3",
    "nvidia-nvtx-cu11==11.8.86",
    "triton==2.2.0",
}
lines = []
for raw in src.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    if line in skip_exact:
        continue
    if any(line.startswith(p) for p in skip_prefixes):
        continue
    lines.append(line)
dst.write_text("\n".join(lines) + "\n")
print(f"kept {len(lines)} dependency lines for docker install")
PY
RUN pip install -r /tmp/requirements.docker.txt

COPY . /workspace/RDPN6D

# Optional optimization used by the project
RUN pip uninstall -y pillow && \
    CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

# Compile local extension
RUN cd core/csrc/fps && python setup.py build_ext --inplace

ENV PYTHONPATH=/workspace/RDPN6D:${PYTHONPATH}

CMD ["/bin/bash"]
