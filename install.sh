#!/usr/bin/env bash
set -euo pipefail

REPO="akhileshsharma99/kryo"
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"
BIN_NAME="kryo"

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Linux) ;;
  *)
    echo "Kryo requires Linux (CRIU is Linux-only). Detected OS: $OS" >&2
    exit 1
    ;;
esac

case "$ARCH" in
  x86_64|amd64)  ARCH_SUFFIX="x64" ;;
  arm64|aarch64) ARCH_SUFFIX="arm64" ;;
  *) echo "Unsupported architecture: $ARCH" >&2; exit 1 ;;
esac

ASSET="kryo-linux-${ARCH_SUFFIX}"
URL="https://github.com/${REPO}/releases/latest/download/${ASSET}"

echo "Downloading ${ASSET}..."
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

if ! curl -fSL "$URL" -o "$TMP"; then
  echo "Download failed. Check https://github.com/${REPO}/releases for available binaries." >&2
  exit 1
fi

chmod +x "$TMP"

if [ -d "$INSTALL_DIR" ] && [ -w "$INSTALL_DIR" ]; then
  mv "$TMP" "${INSTALL_DIR}/${BIN_NAME}"
elif mkdir -p "$INSTALL_DIR" 2>/dev/null && [ -w "$INSTALL_DIR" ]; then
  mv "$TMP" "${INSTALL_DIR}/${BIN_NAME}"
else
  echo "Installing to ${INSTALL_DIR} (requires sudo)..."
  sudo mkdir -p "$INSTALL_DIR"
  sudo mv "$TMP" "${INSTALL_DIR}/${BIN_NAME}"
fi

echo "Installed ${BIN_NAME} to ${INSTALL_DIR}"
"${INSTALL_DIR}/${BIN_NAME}" --version
