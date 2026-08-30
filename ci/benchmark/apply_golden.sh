#!/usr/bin/env bash
# Restore a golden directory onto a stock Lambda GPU image. Not CRIU snapshots.
set -euo pipefail

SRC="${1:?source directory}"
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.cargo/bin:$HOME/.local/bin:${PATH:-}"

if [ ! -f "$SRC/.golden-ok" ]; then
  echo "golden directory missing or incomplete: $SRC" >&2
  exit 1
fi

sudo rsync -a "$SRC/" /
owner="${USER:-ubuntu}"
sudo chown -R "$owner:$owner" \
  "/home/$owner/.cargo" \
  "/home/$owner/.rustup" \
  "/home/$owner/.local" \
  "/home/$owner/.cache" \
  "/home/$owner/kryo/benchmarks/.venv" \
  "/home/$owner/kryo/python/.venv" \
  2>/dev/null || true

echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope >/dev/null || true
sudo nvidia-smi -pm 1 || true
sudo ldconfig || true

if ! command -v criu >/dev/null; then
  echo "criu missing after golden apply" >&2
  exit 1
fi
sudo criu check
if ! command -v cuda-checkpoint >/dev/null; then
  echo "cuda-checkpoint missing after golden apply" >&2
  exit 1
fi
echo "golden apply ok"
