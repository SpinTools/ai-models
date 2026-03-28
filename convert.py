"""
Convert models to ONNX format.
- TensorFlow (.pb) models via tf2onnx
- PyTorch (.pt) models via torch.onnx.export
"""

import os
import subprocess
import tempfile
import requests

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# TensorFlow models to convert
TF_MODELS = [
    {
        "filename": "deeptemp-k16-3.onnx",
        "tf_url": "https://essentia.upf.edu/models/tempo/tempocnn/deeptemp-k16-3.pb",
        "description": "BPM Detector (TempoCNN, 256 BPM classes)",
        "input_names": ["input"],
        "output_names": ["output"],
    },
]

# PyTorch models to convert
PYTORCH_MODELS = [
    {
        "filename": "keynet.onnx",
        "checkpoint_url": "https://raw.githubusercontent.com/a1ex90/MusicalKeyCNN/master/checkpoints/keynet.pt",
        "model_url": "https://raw.githubusercontent.com/a1ex90/MusicalKeyCNN/master/model.py",
        "description": "Key Finder (MusicalKeyCNN, 24 keys)",
        "num_classes": 24,
        "input_shape": (1, 1, 105, 100),  # (batch, channels, freq_bins, time_frames)
    },
]


def download_file(url: str, dest: str, timeout: int = 300) -> None:
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def convert_pb_to_onnx(
    pb_path: str,
    onnx_path: str,
    input_names: list[str],
    output_names: list[str],
) -> None:
    """Convert a TensorFlow frozen graph (.pb) to ONNX using tf2onnx."""
    print("    Converting TF -> ONNX...")

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
        if not os.path.exists(onnx_path):
            raise RuntimeError(
                f"Conversion failed (exit code {result.returncode}): {result.stderr}"
            )

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"    Done ({size_mb:.1f} MB)")


def convert_pytorch_to_onnx(model_config: dict, onnx_path: str) -> None:
    """Convert a PyTorch model to ONNX via torch.onnx.export."""
    import torch
    import importlib.util
    import sys

    print("    Converting PyTorch -> ONNX...")

    # Download model.py to a temp file and import it
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tmp_model:
        print("    Downloading model definition...")
        response = requests.get(model_config["model_url"], timeout=30)
        response.raise_for_status()
        tmp_model.write(response.text)
        tmp_model_path = tmp_model.name

    # Download checkpoint
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_ckpt:
        print("    Downloading checkpoint...")
        download_file(model_config["checkpoint_url"], tmp_ckpt.name)
        tmp_ckpt_path = tmp_ckpt.name

    try:
        # Import the model module dynamically
        spec = importlib.util.spec_from_file_location("keynet_model", tmp_model_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["keynet_model"] = mod
        spec.loader.exec_module(mod)

        # Instantiate and load weights
        model = mod.KeyNet(num_classes=model_config["num_classes"])
        state_dict = torch.load(tmp_ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        # Export to ONNX
        dummy_input = torch.randn(*model_config["input_shape"])
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch", 3: "time_frames"},
                "output": {0: "batch"},
            },
            opset_version=13,
        )

        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"    Done ({size_mb:.1f} MB)")

    finally:
        os.remove(tmp_model_path)
        os.remove(tmp_ckpt_path)
        sys.modules.pop("keynet_model", None)


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- TensorFlow conversions ---
    print("Converting TF models to ONNX...\n")
    for model in TF_MODELS:
        onnx_path = os.path.join(MODELS_DIR, model["filename"])
        if os.path.exists(onnx_path):
            size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            print(f"  Skipping {model['description']} (already exists, {size_mb:.1f} MB)")
            continue

        print(f"  Processing {model['description']}...")
        with tempfile.NamedTemporaryFile(suffix=".pb", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            print(f"    Downloading from {model['tf_url']}...")
            download_file(model["tf_url"], tmp_path)
            size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
            print(f"    Downloaded ({size_mb:.1f} MB)")
            convert_pb_to_onnx(tmp_path, onnx_path, model["input_names"], model["output_names"])
        except Exception as e:
            print(f"  ERROR: {e}")
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # --- PyTorch conversions ---
    print("\nConverting PyTorch models to ONNX...\n")
    for model in PYTORCH_MODELS:
        onnx_path = os.path.join(MODELS_DIR, model["filename"])
        if os.path.exists(onnx_path):
            size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            print(f"  Skipping {model['description']} (already exists, {size_mb:.1f} MB)")
            continue

        print(f"  Processing {model['description']}...")
        try:
            convert_pytorch_to_onnx(model, onnx_path)
        except Exception as e:
            print(f"  ERROR: {e}")
            if os.path.exists(onnx_path):
                os.remove(onnx_path)

    print("\nDone. Models are in:", MODELS_DIR)


if __name__ == "__main__":
    main()
