#!/usr/bin/env bash
# Install a CUDA 12.x vLLM wheel. PyPI's default vllm wants libcudart.so.13.
set -euo pipefail

REPO="${1:-$HOME/kryo}"
cd "$REPO"
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.cargo/bin:$HOME/.local/bin:${PATH:-}"

ARCH="$(uname -m)"
VLLM_VERSION="0.27.1"
# Closest CUDA 12 wheel on current vLLM releases (no cu128 builds).
CUDA_TAG="cu129"
WHEEL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+${CUDA_TAG}-cp38-abi3-manylinux_2_28_${ARCH}.whl"
echo "installing vllm ${VLLM_VERSION}+${CUDA_TAG} from GitHub"
uv pip install --python benchmarks/.venv/bin/python \
  "${WHEEL}" \
  --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
benchmarks/.venv/bin/python - <<'PY'
import torch
import vllm

print(
    f"vllm {vllm.__version__} torch {torch.__version__} "
    f"cuda {torch.version.cuda} available={torch.cuda.is_available()}"
)
if not torch.cuda.is_available():
    raise SystemExit("torch cannot see the GPU after vllm install")
PY
