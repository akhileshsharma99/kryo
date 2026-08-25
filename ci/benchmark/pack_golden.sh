#!/usr/bin/env bash
# Pack Kryo bench tools, venv, and weight caches. Not CRIU GPU snapshots.
set -euo pipefail

OUT="${1:?destination .tgz}"
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.cargo/bin:$HOME/.local/bin:${PATH:-}"

list="$(mktemp)"
cleanup() { rm -f "$list"; }
trap cleanup EXIT

add() {
  local rel="$1"
  if [ -e "/$rel" ]; then
    printf '%s\n' "$rel" >>"$list"
  fi
}

add usr/local/bin/kryo
add usr/local/bin/cuda-checkpoint
add var/lib/kryo-bench
add "home/${USER:-ubuntu}/.cargo"
add "home/${USER:-ubuntu}/.rustup"
add "home/${USER:-ubuntu}/.local"
add "home/${USER:-ubuntu}/.cache/huggingface"
add "home/${USER:-ubuntu}/.cache/uv"
add "home/${USER:-ubuntu}/kryo/benchmarks/.venv"
add "home/${USER:-ubuntu}/kryo/python/.venv"
add "home/${USER:-ubuntu}/kryo/benchmarks/scenarios/yolov8n.pt"

if command -v dpkg >/dev/null; then
  while read -r path; do
    case "$path" in
      /*) printf '%s\n' "${path#/}" >>"$list" ;;
    esac
  done < <(dpkg -L criu 2>/dev/null || true)
fi

if [ ! -s "$list" ]; then
  echo "golden pack list is empty" >&2
  exit 1
fi

sort -u "$list" -o "$list"
tmp="$(mktemp --suffix=.tgz)"
sudo tar -C / --exclude='**/.kryo/snapshots' --exclude='**/__pycache__' \
  --warning=no-file-changed -czf "$tmp" --files-from="$list"
sudo chmod 644 "$tmp"
mkdir -p "$(dirname "$OUT")"
if [[ "$OUT" == /lambda/nfs/* ]]; then
  cp "$tmp" "$OUT"
else
  sudo mv "$tmp" "$OUT"
fi
rm -f "$tmp"
echo "packed golden $OUT ($(du -h "$OUT" | awk '{print $1}'))"
