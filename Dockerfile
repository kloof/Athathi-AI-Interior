# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    git curl wget \
    nodejs npm \
    libgl1-mesa-glx libglib2.0-0 libegl1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /workspace/respace

# Python deps (cached layer — only rebuilds if requirements.txt changes)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install packaging && \
    pip install torch==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# Frontend deps (cached layer — only rebuilds if package.json changes)
COPY src/frontend/package.json src/frontend/package-lock.json src/frontend/
RUN cd src/frontend && npm install

# Copy project source + data/cache + metadata
COPY . .

# Install the project package
RUN pip install -e .

# Install Claude Code
RUN npm install -g @anthropic-ai/claude-code

EXPOSE 5173 8000

# Default: bash so you can run train/eval/serve as needed
CMD ["/bin/bash"]
