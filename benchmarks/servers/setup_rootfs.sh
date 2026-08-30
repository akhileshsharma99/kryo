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

if [ -d "$DEST" ] && { [ -e "$DEST/usr/bin/python3" ] || [ -e "$DEST/usr/local/bin/python3" ] || [ -e "$DEST/opt/tritonserver/bin/tritonserver" ]; }; then
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
  if command -v nvidia-container-cli >/dev/null; then
    sudo nvidia-container-cli --load-kmods configure \
      --ldconfig=@/sbin/ldconfig --device=all --compute --utility "$DEST" \
      && return 0
  fi
  echo "nvidia-container-cli configure failed; copying host driver libs"
  while read -r lib; do
    [ -e "$lib" ] || continue
    rel="${lib#/}"
    sudo mkdir -p "$DEST/$(dirname "$rel")"
    sudo cp -a "$lib" "$DEST/$rel"
  done < <(ldconfig -p | awk -F'=> ' '/libcuda|libnvidia/{print $2}' | sort -u)
}

inject_nvidia || true
echo "rootfs ready $DEST"
ls -ld "$DEST"
