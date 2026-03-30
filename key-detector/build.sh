#!/usr/bin/env bash
#
# Build the key-detector standalone binary using PyInstaller.
# The binary bundles Python, librosa, onnxruntime, and the keynet.onnx model.
#
# Usage: ./build.sh
# Output: dist/detect_key (standalone executable)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_PATH="$SCRIPT_DIR/../models/keynet.onnx"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Error: keynet.onnx not found at $MODEL_PATH"
  echo "Run mirror.py and convert.py first."
  exit 1
fi

cd "$SCRIPT_DIR"

echo "Building key-detector binary..."
echo "  Model: $MODEL_PATH"

# Determine platform suffix
ARCH=$(uname -m)
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
SUFFIX="${OS}-${ARCH}"

pyinstaller \
  --onefile \
  --name "key-detector-${SUFFIX}" \
  --add-data "${MODEL_PATH}:." \
  --distpath dist \
  --workpath build \
  --specpath build \
  --clean \
  --noconfirm \
  detect_key.py

BINARY="dist/key-detector-${SUFFIX}"
echo ""
echo "Built: $BINARY"
echo "Size: $(du -h "$BINARY" | cut -f1)"
echo ""
echo "Test: $BINARY /path/to/track.mp3"
