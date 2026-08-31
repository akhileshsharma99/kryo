#!/usr/bin/env bash
# Bootstrap a Lambda Ubuntu GPU VM for Kryo with/without benchmarks.
set -euo pipefail

REPO="${1:-$HOME/kryo}"
cd "$REPO"

export DEBIAN_FRONTEND=noninteractive
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.cargo/bin:$HOME/.local/bin:${PATH:-}"

if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi not found; this image has no NVIDIA driver" >&2
  exit 1
fi

DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | cut -d. -f1)"
if [ "${DRIVER:-0}" -lt 550 ]; then
  echo "NVIDIA driver ${DRIVER} is older than 550 (required by cuda-checkpoint)" >&2
  exit 1
fi

sudo nvidia-smi -pm 1 || true
echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope >/dev/null || true

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  curl \
  git \
  python3 \
  software-properties-common

if ! command -v criu >/dev/null || ! criu check >/dev/null 2>&1; then
  sudo add-apt-repository -y ppa:criu/ppa
  sudo apt-get update
  sudo apt-get install -y criu
fi
sudo criu check

if ! command -v cuda-checkpoint >/dev/null; then
  tmp="$(mktemp -d)"
  git clone --depth 1 https://github.com/NVIDIA/cuda-checkpoint.git "$tmp/cuda-checkpoint"
  arch="$(uname -m)"
  case "$arch" in
    x86_64) bin="$tmp/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint" ;;
    aarch64) bin="$tmp/cuda-checkpoint/bin/aarch64_Linux/cuda-checkpoint" ;;
    *) bin="" ;;
  esac
  if [ -z "$bin" ] || [ ! -x "$bin" ]; then
    echo "cuda-checkpoint binary not found for $arch" >&2
    exit 1
  fi
  sudo install -m 755 "$bin" /usr/local/bin/cuda-checkpoint
  rm -rf "$tmp"
fi
cuda-checkpoint --help >/dev/null

if ! command -v rustc >/dev/null || ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
if [ -f "$HOME/.cargo/env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

cargo build --release --locked
sudo install -m 755 target/release/kryo /usr/local/bin/kryo
kryo --version

uv python install 3.11
uv sync --directory python
uv sync --directory benchmarks

# uv sync pulls CPU/default torch first; replace it with a wheel that matches
# the driver's CUDA (Lambda A10 currently reports 12.8).
CUDA_MM="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n1)"
case "$CUDA_MM" in
  12.8*|12.9*|13.*) TORCH_CUDA=cu128 ;;
  12.6*|12.7*) TORCH_CUDA=cu126 ;;
  12.4*|12.5*) TORCH_CUDA=cu124 ;;
  *) TORCH_CUDA=cu121 ;;
esac
echo "installing torch for CUDA ${CUDA_MM:-unknown} ($TORCH_CUDA)"
uv pip uninstall --python benchmarks/.venv/bin/python -y torch torchvision triton || true
uv pip install --python benchmarks/.venv/bin/python \
  torch torchvision --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
benchmarks/.venv/bin/python - <<'PY'
import torch
print(f"torch {torch.__version__} cuda {torch.version.cuda} available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("torch cannot see the GPU after CUDA wheel install")
PY

# Use the venv directly. `uv run` would re-sync the lockfile and replace CUDA torch.
benchmarks/.venv/bin/python benchmarks/download_models.py

sudo mkdir -p /var/lib/kryo-bench
echo "$(date -Iseconds) $(kryo --version)" | sudo tee /var/lib/kryo-bench/golden.stamp >/dev/null

echo "setup complete"
