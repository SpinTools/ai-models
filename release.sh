#!/usr/bin/env bash
#
# End-to-end release workflow for SpinTools AI Models.
#
# Usage: ./release.sh v2
#
# Steps:
#   1. Download/convert any new or updated models
#   2. Generate manifest.json with URLs pointing to this release tag
#   3. Create GitHub release and upload all .onnx files
#   4. Commit and push manifest.json + models.yaml

set -euo pipefail

TAG="${1:?Usage: ./release.sh <tag>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== SpinTools AI Models Release: $TAG ==="
echo ""

# 1. Ensure models are downloaded/converted
echo "--- Step 1: Mirror ONNX models ---"
python mirror.py
echo ""

echo "--- Step 2: Convert models ---"
python convert.py
echo ""

# 2. Generate manifest
echo "--- Step 3: Generate manifest.json ---"
python generate_manifest.py --tag "$TAG"
echo ""

# 3. Upload release
echo "--- Step 4: Upload to GitHub release ---"
./upload_release.sh "$TAG"
echo ""

# 4. Commit and push
echo "--- Step 5: Commit and push ---"
git add manifest.json models.yaml
if git diff --cached --quiet; then
  echo "No changes to commit."
else
  git commit -m "Release $TAG"
  git push
fi

echo ""
echo "=== Release $TAG complete ==="
echo "Manifest: https://raw.githubusercontent.com/SpinTools/ai-models/main/manifest.json"
