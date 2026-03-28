"""
Convert TensorFlow (.pb) models to ONNX format.
Used for Essentia models that don't have official ONNX releases.
"""

import os
import subprocess
import tempfile
import requests

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

CONVERT_MODELS = [
    {
        "filename": "deeptemp-k16-3.onnx",
        "tf_url": "https://essentia.upf.edu/models/tempo/tempocnn/deeptemp-k16-3.pb",
        "description": "BPM Detector (TempoCNN, 256 BPM classes)",
        "input_names": ["input"],
        "output_names": ["output"],
    },
]


def download_tf_model(url: str, dest: str) -> None:
    print(f"    Downloading TF model from {url}...")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"    Downloaded ({size_mb:.1f} MB)")


def convert_pb_to_onnx(
    pb_path: str,
    onnx_path: str,
    input_names: list[str],
    output_names: list[str],
) -> None:
    """Convert a TensorFlow frozen graph (.pb) to ONNX using tf2onnx."""
    print(f"    Converting to ONNX...")

    cmd = [
        "python", "-m", "tf2onnx.convert",
        "--graphdef", pb_path,
        "--output", onnx_path,
        "--inputs", ",".join(f"{n}:0" for n in input_names),
        "--outputs", ",".join(f"{n}:0" for n in output_names),
        "--opset", "13",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    tf2onnx stderr: {result.stderr}")
        # tf2onnx often prints warnings to stderr but still succeeds
        if not os.path.exists(onnx_path):
            raise RuntimeError(
                f"Conversion failed (exit code {result.returncode}): {result.stderr}"
            )

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"    Converted ({size_mb:.1f} MB)")


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Converting TF models to ONNX...\n")

    for model in CONVERT_MODELS:
        onnx_path = os.path.join(MODELS_DIR, model["filename"])

        if os.path.exists(onnx_path):
            size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            print(f"  Skipping {model['description']} (already exists, {size_mb:.1f} MB)")
            continue

        print(f"  Processing {model['description']}...")

        with tempfile.NamedTemporaryFile(suffix=".pb", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            download_tf_model(model["tf_url"], tmp_path)
            convert_pb_to_onnx(
                tmp_path,
                onnx_path,
                model["input_names"],
                model["output_names"],
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    print("\nDone. Models are in:", MODELS_DIR)


if __name__ == "__main__":
    main()
