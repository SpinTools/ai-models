#!/usr/bin/env bash
#
# Upload all ONNX models in models/ as GitHub release assets.
#
# Usage: ./upload_release.sh [tag]
#   tag defaults to "v1"

set -euo pipefail

TAG="${1:-v1}"
MODELS_DIR="$(dirname "$0")/models"

if [ ! -d "$MODELS_DIR" ] || [ -z "$(ls -A "$MODELS_DIR"/*.onnx 2>/dev/null)" ]; then
  echo "No .onnx files found in $MODELS_DIR"
  echo "Run 'python mirror.py' and 'python convert.py' first."
  exit 1
fi

echo "Creating release $TAG..."

# Create the release (or skip if it exists)
if gh release view "$TAG" &>/dev/null; then
  echo "Release $TAG already exists. Uploading assets..."
else
  gh release create "$TAG" \
    --title "AI Models $TAG" \
    --notes "ONNX audio analysis models for SpinTools.

Models are sourced from Essentia (UPF) and HuggingFace.
See README.md for details on each model." \
    --repo SpinTools/ai-models
fi

# Upload each model
for model in "$MODELS_DIR"/*.onnx; do
  filename=$(basename "$model")
  echo "  Uploading $filename..."
  gh release upload "$TAG" "$model" --clobber --repo SpinTools/ai-models
done

echo ""
echo "Done! Models available at:"
echo "  https://github.com/SpinTools/ai-models/releases/download/$TAG/"
