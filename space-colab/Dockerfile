# GPU base for Spaces. Pinned CUDA so torch wheels match.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/hf_cache \
    TRANSFORMERS_CACHE=/app/hf_cache \
    PORT=7860 \
    SUSPECT_X_DESCRIPTIONS=/app/descriptions

# System Python and build tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip git build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && python -m pip install --upgrade pip

WORKDIR /app

# Pin torch to a CUDA 12.1 build. Order matters — install torch first so
# unsloth/bitsandbytes don't drag in a different CUDA wheel.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.0 \
        torchvision==0.19.0

# Core HF stack. Pinned to avoid resolver thrash within the build window.
RUN pip install \
        "transformers>=4.46,<4.50" \
        "trl>=0.16,<0.20" \
        "peft>=0.13,<0.16" \
        "accelerate>=1.1,<1.5" \
        "datasets>=2.20,<3.5" \
        "bitsandbytes==0.44.1" \
        "sentencepiece" \
        "protobuf" \
        "matplotlib" \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.27" \
        "pydantic>=2.6" \
        "httpx"

# Unsloth dropped: incompatible with trl >=0.16's SFTTrainer signature.
# Plain transformers + peft + bitsandbytes is enough for 4-bit Qwen + LoRA + GRPO.

# Application code.
COPY app.py        /app/app.py
COPY train.py      /app/train.py
COPY server        /app/server
COPY training      /app/training
COPY descriptions  /app/descriptions

# Pre-create dirs the app writes to (on the writable HF Space volume).
RUN mkdir -p /app/out /app/hf_cache && chmod -R 777 /app/out /app/hf_cache

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
