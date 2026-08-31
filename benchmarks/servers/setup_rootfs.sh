#!/usr/bin/env bash
# Pull an official image and unpack it to a stable host path Kryo/CRIU can restore.
# Restore inside Docker overlayfs is not supported (see ci/benchmark/README.md).
set -euo pipefail

IMAGE="${1:?image}"
DEST="${2:?destination dir}"
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

if ! command -v docker >/dev/null; then
  echo "installing docker"
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "${USER:-ubuntu}" || true
  sudo systemctl enable --now docker || sudo service docker start || true
fi

if ! command -v nvidia-container-cli >/dev/null && ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
  echo "installing nvidia-container-toolkit"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker || true
  sudo systemctl restart docker || sudo service docker restart || true
fi

if [ -n "${NGC_API_KEY:-}" ] && [[ "$IMAGE" == nvcr.io/* ]]; then
  echo "$NGC_API_KEY" | sudo docker login nvcr.io -u '$oauthtoken' --password-stdin
fi

echo "pulling $IMAGE"
sudo docker pull "$IMAGE"

already=false
if sudo test -x "$DEST/usr/bin/python3.12" || sudo test -x "$DEST/usr/bin/python3.10"; then
  already=true
fi
if sudo test -s "$DEST/usr/bin/python3" && sudo test -x "$DEST/usr/bin/python3"; then
  already=true
fi
if sudo test -x "$DEST/opt/tritonserver/bin/tritonserver"; then
  already=true
fi
if [ -d "$DEST" ] && [ "$already" = true ]; then
  echo "rootfs already present $DEST"
else
  echo "exporting $IMAGE -> $DEST"
  sudo rm -rf "$DEST"
  sudo mkdir -p "$DEST"
  cid="$(sudo docker create "$IMAGE")"
  sudo docker export "$cid" | sudo tar -C "$DEST" -xf -
  sudo docker rm "$cid" >/dev/null
fi

for d in proc sys dev dev/shm tmp run root/.cache/huggingface; do
  sudo mkdir -p "$DEST/$d"
done

inject_nvidia() {
  echo "copying host NVIDIA driver libs into rootfs"
  while read -r lib; do
    [ -e "$lib" ] || continue
    rel="${lib#/}"
    sudo mkdir -p "$DEST/$(dirname "$rel")"
    sudo cp -a "$lib" "$DEST/$rel"
  done < <(ldconfig -p | awk -F'=> ' '/libcuda|libnvidia/{print $2}' | sort -u)
}

inject_nvidia || true
# nvidia-container-cli can replace /usr/bin/python3 with a 0-byte stub and
# leave /usr mode 0700. Restore a real interpreter and make the tree walkable.
sudo chmod 755 "$DEST" "$DEST/usr" "$DEST/usr/bin" "$DEST/usr/lib" "$DEST/opt" 2>/dev/null || true
if ! sudo test -x "$DEST/usr/bin/python3" || ! sudo test -s "$DEST/usr/bin/python3"; then
  if sudo test -x "$DEST/usr/bin/python3.12"; then
    sudo ln -sfn python3.12 "$DEST/usr/bin/python3"
  elif sudo test -x "$DEST/usr/bin/python3.10"; then
    sudo ln -sfn python3.10 "$DEST/usr/bin/python3"
  fi
fi
echo "rootfs ready $DEST"
ls -ld "$DEST"
