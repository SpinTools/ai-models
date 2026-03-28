"""
Download existing ONNX models from Essentia and HuggingFace into models/ directory.
These models are already in ONNX format and just need to be mirrored.
"""

import os
import requests

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

MIRROR_MODELS = [
    {
        "filename": "discogs-effnet-bsdynamic-1.onnx",
        "url": "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bsdynamic-1.onnx",
        "description": "Genre Tagger (EfficientNet, 400 Discogs labels)",
    },
    {
        "filename": "emomusic-audioset-vggish-2.onnx",
        "url": "https://essentia.upf.edu/models/classification-heads/emomusic/emomusic-audioset-vggish-2.onnx",
        "description": "Energy Level (VGGish, arousal regression)",
    },
    {
        "filename": "emomusic-audioset-vggish-1.onnx",
        "url": "https://essentia.upf.edu/models/classification-heads/emomusic/emomusic-audioset-vggish-1.onnx",
        "description": "Mood (VGGish, valence regression)",
    },
    {
        "filename": "danceability-audioset-vggish-1.onnx",
        "url": "https://essentia.upf.edu/models/classification-heads/danceability/danceability-audioset-vggish-1.onnx",
        "description": "Danceability (VGGish, binary classification)",
    },
    {
        "filename": "discogs-maest-30s-pw-1.onnx",
        "url": "https://essentia.upf.edu/models/feature-extractors/maest/discogs-maest-30s-pw-1.onnx",
        "description": "Genre Tagger Pro (MAEST transformer, 400 labels)",
    },
]


def download_file(url: str, dest: str, description: str) -> None:
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"  Skipping {description} (already exists, {size_mb:.1f} MB)")
        return

    print(f"  Downloading {description}...")
    print(f"    URL: {url}")

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = (downloaded / total) * 100
                print(f"\r    {downloaded / (1024*1024):.1f} / {total / (1024*1024):.1f} MB ({pct:.0f}%)", end="")

    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"\n    Done ({size_mb:.1f} MB)")


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Mirroring ONNX models...\n")

    for model in MIRROR_MODELS:
        dest = os.path.join(MODELS_DIR, model["filename"])
        try:
            download_file(model["url"], dest, model["description"])
        except Exception as e:
            print(f"  ERROR downloading {model['filename']}: {e}")

    print("\nDone. Models are in:", MODELS_DIR)


if __name__ == "__main__":
    main()
