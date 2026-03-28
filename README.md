# SpinTools AI Models

AI-powered audio analysis models for [SpinTools](https://spintools.app), a music management app built for DJs. These models run locally on your computer to analyze your music library — tagging genre, detecting BPM, finding musical key, scoring energy and mood, and more.

Models are distributed as [ONNX](https://onnx.ai/) files via GitHub Releases so the SpinTools app can download them on demand without bundling large files in the installer.

## Available Models

| Model | What it does | Architecture | Size |
|-------|-------------|--------------|------|
| **Genre Tagger** | Automatically tags genre from 400+ styles (Deep House, Drum & Bass, Trap, etc.) | EfficientNet (Discogs) | ~17 MB |
| **Energy Level** | Scores tracks from low to high energy for planning set flow | VGGish (AudioSet) | ~52 KB |
| **Mood** | Tags tracks as dark/moody or bright/uplifting | VGGish (AudioSet) | ~3 KB |
| **Danceability** | Rates how danceable a track is based on groove and rhythm | VGGish (AudioSet) | ~52 KB |
| **BPM Detector** | Finds tempo — handles breakbeats and syncopated patterns | TempoCNN | ~1.2 MB |
| **Genre Tagger Pro** | Advanced genre tagger with 500+ styles, better on long build-ups | MAEST Transformer | ~332 MB |

All models run entirely on your machine using [ONNX Runtime](https://onnxruntime.ai/) (WebAssembly backend). No audio is sent to any server.

## Download URLs

Models are available as release assets:

```
https://github.com/SpinTools/ai-models/releases/download/v1/{filename}.onnx
```

The SpinTools app downloads models from these URLs when you install them from the AI Tools page.

---

## For Contributors: Adding and Converting Models

This repo also contains scripts for sourcing and preparing models. Models come from two places:

- **Essentia** ([essentia.upf.edu/models](https://essentia.upf.edu/models.html)) — music analysis models from the Music Technology Group at Universitat Pompeu Fabra (Barcelona)

Some models are already published in ONNX format and just need to be mirrored. Others are only available as TensorFlow frozen graphs (`.pb`) and need conversion to ONNX.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Mirror existing ONNX models

Downloads models that are already in ONNX format from their original sources into `models/`:

```bash
python mirror.py
```

### Convert TensorFlow models to ONNX

Downloads TensorFlow `.pb` models and converts them to ONNX using [tf2onnx](https://github.com/onnx/tensorflow-onnx):

```bash
python convert.py
```

### Upload to GitHub Releases

Creates (or updates) a GitHub release and uploads all `.onnx` files from `models/`:

```bash
./upload_release.sh v1
```

Requires the [GitHub CLI](https://cli.github.com/) (`gh`) to be installed and authenticated.

### Adding a new model

1. If the model is already ONNX, add it to the `MIRROR_MODELS` list in `mirror.py`
2. If the model is TensorFlow (`.pb`), add it to the `CONVERT_MODELS` list in `convert.py` with the correct input/output tensor names
3. Run the appropriate script, then `upload_release.sh` to publish
4. Update the seed data in the SpinTools app (`electron/src/services/ModelManagerService.ts`) with the new model entry

## Licensing

Models are sourced from third-party research projects. Refer to each model's original repository for license terms:

- [Essentia Models](https://essentia.upf.edu/models.html) — AGPL-3.0
