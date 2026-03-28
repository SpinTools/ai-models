"""
Generate manifest.json from models.yaml for a given release tag.

Usage:
    python generate_manifest.py --tag v2
    python generate_manifest.py --tag v1 --models-dir ./models
"""

import argparse
import json
import os

import yaml

REPO = "SpinTools/ai-models"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MANIFEST_SCHEMA_VERSION = 1


def build_download_url(tag: str, filename: str) -> str:
    return f"https://github.com/{REPO}/releases/download/{tag}/{filename}"


def get_file_size(filename: str, models_dir: str) -> int | None:
    path = os.path.join(models_dir, filename)
    if os.path.exists(path):
        return os.path.getsize(path)
    return None


def generate(tag: str, models_dir: str) -> dict:
    yaml_path = os.path.join(os.path.dirname(__file__), "models.yaml")

    with open(yaml_path, "r") as f:
        models = yaml.safe_load(f)

    manifest_models = []
    for m in models:
        file_size = get_file_size(m["filename"], models_dir)

        entry = {
            "slug": m["slug"],
            "name": m["name"],
            "description": m["description"],
            "version": m["version"],
            "category": m["category"],
            "downloadUrl": build_download_url(tag, m["filename"]),
            "fileSize": file_size,
            "outputConfig": m["outputConfig"],
            "defaultOutputConfig": m["outputConfig"],
            "preprocessingConfig": m["preprocessingConfig"],
        }
        manifest_models.append(entry)

    return {
        "schemaVersion": MANIFEST_SCHEMA_VERSION,
        "releaseTag": tag,
        "models": manifest_models,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate manifest.json from models.yaml"
    )
    parser.add_argument(
        "--tag", required=True, help="GitHub release tag (e.g., v1, v2)"
    )
    parser.add_argument(
        "--models-dir",
        default=MODELS_DIR,
        help="Directory containing .onnx files (for file size calculation)",
    )
    args = parser.parse_args()

    manifest = generate(args.tag, args.models_dir)

    manifest_path = os.path.join(os.path.dirname(__file__), "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated manifest.json for release {args.tag}")
    print(f"  Models: {len(manifest['models'])}")
    for m in manifest["models"]:
        size = f"{m['fileSize'] / (1024*1024):.1f} MB" if m["fileSize"] else "unknown size"
        print(f"    {m['slug']} v{m['version']} ({size})")


if __name__ == "__main__":
    main()
