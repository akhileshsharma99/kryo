#!/usr/bin/env bash
# Install a CUDA 12.x vLLM wheel. PyPI's default vllm wants libcudart.so.13.
set -euo pipefail

REPO="${1:-$HOME/kryo}"
cd "$REPO"
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.cargo/bin:$HOME/.local/bin:${PATH:-}"

# flashinfer JIT-compiles sampling kernels on first generate; vLLM then
# calls `ninja`. Golden images from older setup.sh do not have it.
if ! command -v ninja >/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends ninja-build
fi

ARCH="$(uname -m)"
VLLM_VERSION="0.27.1"
# Closest CUDA 12 wheel on current vLLM releases (no cu128 builds).
CUDA_TAG="cu129"
WHEEL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+${CUDA_TAG}-cp38-abi3-manylinux_2_28_${ARCH}.whl"
echo "installing vllm ${VLLM_VERSION}+${CUDA_TAG} from GitHub"
uv pip install --python benchmarks/.venv/bin/python \
  "${WHEEL}" \
  --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
# vLLM 0.27 pulls flashinfer 0.6.16.post2/post3; those crash on Python 3.11
# (`array.array[int]`). post4 restores the import.
uv pip install --python benchmarks/.venv/bin/python \
  "flashinfer-python>=0.6.16.post4"
benchmarks/.venv/bin/python - <<'PY'
import torch
import vllm

print(
    f"vllm {vllm.__version__} torch {torch.__version__} "
    f"cuda {torch.version.cuda} available={torch.cuda.is_available()}"
)
if not torch.cuda.is_available():
    raise SystemExit("torch cannot see the GPU after vllm install")
import flashinfer.comm  # noqa: F401
print("flashinfer.comm import ok")
PY
