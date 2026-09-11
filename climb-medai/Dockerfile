FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# Build context: climb-medai/
# run scripts/fetch_weights.sh before `docker build`.

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    NVIDIA_DRIVER_CAPABILITIES=all \
    NVIDIA_VISIBLE_DEVICES=all \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="8.9;12.0" \
    HOME=/tmp \
    TORCH_HOME=/tmp/torch_cache \
    XDG_CACHE_HOME=/tmp/xdg_cache \
    TRITON_CACHE_DIR=/tmp/triton_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl build-essential software-properties-common \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.9 python3.9-venv python3.9-distutils python3.9-dev \
    && rm -rf /var/lib/apt/lists/*

# python3.9
RUN curl -sS https://bootstrap.pypa.io/pip/3.9/get-pip.py | python3.9 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Load-bearing pins: torch/torchvision 2.8.0+cu128, triton 3.4.0 (mamba fork + sm_120).
RUN python3.9 -m pip install --no-cache-dir \
        torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128 \
    && python3.9 -m pip install --no-cache-dir \
        triton==3.4.0 \
        opencv-python-headless==5.0.0.93 \
        numpy==2.0.2 \
        scipy==1.13.1 \
        PyYAML==6.0.3 \
        packaging==26.2 \
        ninja==1.13.0

# Vendored mamba_ssm fork
COPY Thirdparty/mamba /opt/Thirdparty/mamba
RUN MAMBA_SKIP_CUDA_BUILD=TRUE python3.9 -m pip install --no-cache-dir --no-build-isolation \
    /opt/Thirdparty/mamba

WORKDIR /opt/climb-medai
COPY colonstreamsfsnet colonstreamsfsnet
COPY src src
COPY configs configs
COPY scripts/bench_depth.py scripts/bench_depth.py
COPY run.py run.py

# Checkpoint for Depth Estimation (ColonStreamSfSNet) weights.
COPY weights weights
RUN echo "754c3b986bb2ca2499bde9c127a3ce30c5aa363b5e2fce3683c5f833fcd76b02  /opt/climb-medai/weights/colonstreamsfsnet/best_model.pth" \
    | sha256sum -c -

# Build-time warm-up: fetch RAFT weights (no network at run time) and trigger the
# Mamba2 Triton JIT. Depth pass is skipped when no GPU is visible (no CPU path).
RUN mkdir -p /tmp/torch_cache /tmp/xdg_cache /tmp/triton_cache \
    && chmod -R 1777 /tmp/torch_cache /tmp/xdg_cache /tmp/triton_cache \
    && python3.9 -c "from torchvision.models.optical_flow import raft_small, Raft_Small_Weights; raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False)" \
    && ( python3.9 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
         && python3.9 scripts/bench_depth.py --n 2 --warmup 1 --device cuda:0 \
         || echo "WARNING: ColonStreamSfSNet warm-up skipped -- no GPU visible during docker build." ) \
    && chmod -R 1777 /tmp/torch_cache /tmp/xdg_cache /tmp/triton_cache

ENTRYPOINT ["python3.9", "/opt/climb-medai/run.py"]
# Scorer invokes the image with no args; the operating point is whatever
# configs/climb_v9c.yaml specifies (run.py has no CLI tuning flags anymore).
CMD ["/input", "/output"]
