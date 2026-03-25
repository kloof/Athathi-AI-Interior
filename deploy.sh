#!/bin/bash
set -e

IMAGE_NAME="${RESPACE_IMAGE:-respace}"
IMAGE_TAG="${RESPACE_TAG:-latest}"
REGISTRY="${RESPACE_REGISTRY:-}"  # e.g. docker.io/username or ghcr.io/username

FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
if [ -n "$REGISTRY" ]; then
    FULL_IMAGE="${REGISTRY}/${FULL_IMAGE}"
fi

usage() {
    echo "Usage: ./deploy.sh [build|test|push|pull|run]"
    echo ""
    echo "Environment variables:"
    echo "  RESPACE_REGISTRY  - Registry prefix (e.g. docker.io/user, ghcr.io/user)"
    echo "  RESPACE_IMAGE     - Image name (default: respace)"
    echo "  RESPACE_TAG       - Image tag (default: latest)"
    echo ""
    echo "Examples:"
    echo "  # Build locally"
    echo "  ./deploy.sh build"
    echo ""
    echo "  # Push to Docker Hub"
    echo "  RESPACE_REGISTRY=docker.io/myuser ./deploy.sh push"
    echo ""
    echo "  # Pull and run on new instance"
    echo "  RESPACE_REGISTRY=docker.io/myuser ./deploy.sh pull"
    echo "  RESPACE_REGISTRY=docker.io/myuser ./deploy.sh run"
    exit 1
}

cmd_build() {
    echo "Building ${FULL_IMAGE} ..."
    docker build -t "$FULL_IMAGE" .
    echo "Done. Image: ${FULL_IMAGE}"
}

cmd_test() {
    echo "Testing ${FULL_IMAGE} ..."
    echo "--- Image size ---"
    docker image ls "$FULL_IMAGE" --format "{{.Size}}"
    echo "--- Smoke test: Python imports ---"
    docker run --rm "$FULL_IMAGE" python3 -c "
import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}')
import transformers; print(f'transformers={transformers.__version__}')
import vllm; print(f'vllm={vllm.__version__}')
from src.dataset import *; print('dataset OK')
from src.sample import *; print('sample OK')
from src.utils import *; print('utils OK')
from src.respace import *; print('respace OK')
import os
for k in ['PTH_ASSETS_EMBED','PTH_ASSETS_METADATA','PTH_DATASET_CACHE','PTH_INVALID_ROOMS']:
    v = os.getenv(k, 'NOT SET')
    exists = os.path.exists(v) if v != 'NOT SET' else False
    print(f'{k}={v}  exists={exists}')
print('All checks passed.')
" --env-file .env
    echo "--- Smoke test: Claude Code ---"
    docker run --rm "$FULL_IMAGE" claude --version
    echo "--- Smoke test: Frontend ---"
    docker run --rm "$FULL_IMAGE" bash -c "cd src/frontend && ls node_modules/.package-lock.json && echo 'node_modules OK'"
    echo "All tests passed."
}

cmd_push() {
    echo "Pushing ${FULL_IMAGE} ..."
    docker push "$FULL_IMAGE"
    echo "Done."
}

cmd_pull() {
    echo "Pulling ${FULL_IMAGE} ..."
    docker pull "$FULL_IMAGE"
    echo "Done."
}

cmd_run() {
    echo "Running ${FULL_IMAGE} ..."
    docker run -it --gpus all \
        --env-file .env \
        -v "${HOST_3DFRONT:-/data/3D-FRONT}:/data/3D-FRONT" \
        -v "${HOST_3DFUTURE:-/data/3D-FUTURE-assets}:/data/3D-FUTURE-assets" \
        -v "${HOST_STAGE2:-/data/stage-2-dedup}:/data/stage-2-dedup" \
        -p 5173:5173 \
        -p 8000:8000 \
        "$FULL_IMAGE"
}

case "${1:-}" in
    build) cmd_build ;;
    test)  cmd_test ;;
    push)  cmd_push ;;
    pull)  cmd_pull ;;
    run)   cmd_run ;;
    *)     usage ;;
esac
