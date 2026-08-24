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

if ! command -v criu >/dev/null; then
  sudo add-apt-repository -y ppa:criu/ppa
  sudo apt-get update
  sudo apt-get install -y criu
fi
sudo criu check

if ! command -v cuda-checkpoint >/dev/null; then
  tmp="$(mktemp -d)"
  git clone --depth 1 https://github.com/NVIDIA/cuda-checkpoint.git "$tmp/cuda-checkpoint"
  make -C "$tmp/cuda-checkpoint" || true
  bin="$(find "$tmp/cuda-checkpoint" -type f -name cuda-checkpoint | head -n1)"
  if [ -z "$bin" ]; then
    echo "cuda-checkpoint binary not found after build" >&2
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
uv pip install --python benchmarks/.venv/bin/python \
  torch torchvision --index-url https://download.pytorch.org/whl/cu124

uv run --directory benchmarks python download_models.py

echo "setup complete"
