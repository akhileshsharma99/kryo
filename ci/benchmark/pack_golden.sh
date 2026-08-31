#!/usr/bin/env bash
# Copy Kryo bench tools and venvs onto a raw NFS directory.
# No tar, no gzip. Hugging Face weights stay on the VM (extra_weights).
set -euo pipefail

OUT="${1:?destination directory}"
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.cargo/bin:$HOME/.local/bin:${PATH:-}"

list="$(mktemp)"
cleanup() { rm -f "$list"; }
trap cleanup EXIT

add() {
  local rel="$1"
  case "$rel" in
    ""|.|proc|proc/*|sys|sys/*|dev|dev/*|run|run/*) return ;;
  esac
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
# Hugging Face weights are downloaded per job. Copying 32B onto NFS with
# tar/gzip blocked the actual benches for 15+ minutes.
add "home/${USER:-ubuntu}/.cache/uv"
add "home/${USER:-ubuntu}/kryo/benchmarks/.venv"
add "home/${USER:-ubuntu}/kryo/python/.venv"
add "home/${USER:-ubuntu}/kryo/benchmarks/scenarios/yolov8n.pt"

if command -v dpkg >/dev/null; then
  # criu's ELF deps live in other packages; dpkg -L criu alone is not enough.
  for pkg in criu libprotobuf-c1 libnet1; do
    while read -r path; do
      case "$path" in
        /|/.|/proc|/proc/*|/sys|/sys/*|/dev|/dev/*|/run|/run/*) continue ;;
        /*) add "${path#/}" ;;
      esac
    done < <(dpkg -L "$pkg" 2>/dev/null || true)
  done
fi

if [ ! -s "$list" ]; then
  echo "golden pack list is empty" >&2
  exit 1
fi

sort -u "$list" -o "$list"
lock_dir="$(dirname "$OUT")/.lock-$(basename "$OUT")"
waited=0
while ! sudo mkdir "$lock_dir" 2>/dev/null; do
  if [ -f "$OUT/.golden-ok" ]; then
    echo "golden already packed $OUT"
    exit 0
  fi
  waited=$((waited + 5))
  if [ "$waited" -ge 3600 ]; then
    echo "timeout waiting for golden pack lock $lock_dir" >&2
    exit 1
  fi
  sleep 5
done
release_lock() { sudo rmdir "$lock_dir" 2>/dev/null || true; }
trap 'release_lock; cleanup' EXIT
if [ -f "$OUT/.golden-ok" ]; then
  echo "golden already packed $OUT"
  exit 0
fi
sudo rm -rf "$OUT"
sudo mkdir -p "$OUT"
# Raw directory copy. No tar, no gzip.
while read -r rel; do
  src="/$rel"
  [ -e "$src" ] || continue
  dest="$OUT/$rel"
  sudo mkdir -p "$(dirname "$dest")"
  if [ -d "$src" ]; then
    sudo rsync -a --exclude '.kryo/snapshots' --exclude '__pycache__' "$src/" "$dest/"
  else
    sudo rsync -a "$src" "$dest"
  fi
done <"$list"
size="$(sudo du -sb "$OUT" | awk '{print $1}')"
if [ "${size:-0}" -lt 50000000 ]; then
  echo "golden too small (${size:-0} bytes); refusing to stamp a stub image" >&2
  sudo rm -rf "$OUT"
  exit 1
fi
echo ok | sudo tee "$OUT/.golden-ok" >/dev/null
echo "packed golden $OUT ($(sudo du -sh "$OUT" | awk '{print $1}'))"
