"""
SpinTools AI Engine

Unified inference engine for all SpinTools AI models.
Runs as a long-lived server process, receiving commands via stdin
and returning results via stdout.

Usage:
    spintools-ai --server --models-dir /path/to/models
    spintools-ai --run <model_slug> --model-path /path/to/model.onnx --file /path/to/track.mp3

Server commands (stdin):
    LOAD <slug> <model_path> [vggish_path]
    RUN <slug> <audio_path>
    UNLOAD <slug>
    QUIT

Server responses (stdout):
    READY
    LOADED <slug>
    RESULT <slug> <value>
    ERROR <slug> <message>
"""

import sys
import os
import warnings
import json

# Suppress all warnings (librosa, soundfile, etc.)
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import librosa
import onnxruntime as ort


# ---------------------------------------------------------------------------
# Model preprocessing configurations
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "musicalkeycnn-v1": {
        "type": "cqt_classification",
        "sr": 44100,
        "duration": 30,
        "cqt": {
            "hop_length": 8820,
            "n_bins": 105,
            "bins_per_octave": 24,
            "fmin": 65,
        },
        "post": "camelot_key",
    },
    "discogs-effnet-bs64": {
        "type": "mel_classification",
        "sr": 16000,
        "duration": 30,
        "mel": {
            "n_mels": 96,
            "hop_length": 512,
            "n_fft": 2048,
            "fmin": 0,
            "fmax": 8000,
        },
        "tensor_format": "NCHW",  # [1, 1, n_mels, n_frames]
        "labels_file": "labels.json",
    },
    "discogs-maest-30s-pw-519l": {
        "type": "mel_classification",
        "sr": 16000,
        "duration": 30,
        "mel": {
            "n_mels": 96,
            "hop_length": 256,
            "n_fft": 512,
            "fmin": 0,
            "fmax": 8000,
        },
        "tensor_format": "NTM",  # [1, n_frames, n_mels]
        "target_frames": 1876,
        "labels_file": "labels.json",
        "log_compress": "essentia",  # log10(1 + mel * 10000) + z-norm
    },
    "tempocnn-deeptemp-k16": {
        "type": "mel_classification",
        "sr": 11025,
        "duration": 30,
        "mel": {
            "n_mels": 40,
            "hop_length": 512,
            "n_fft": 1024,
            "fmin": 30,
            "fmax": 5500,
        },
        "tensor_format": "NHWC",  # [1, n_mels, n_frames, 1]
        "post": "bpm",  # 30 + argmax
    },
    "arousal-regression-audioset-vggish": {
        "type": "vggish_head",
        "output_index": 1,  # arousal
        "post": "scale_1_9_to_1_10",
    },
    "valence-regression-audioset-vggish": {
        "type": "vggish_head",
        "output_index": 0,  # valence
        "post": "mood_label",
    },
    "danceability-audioset-vggish": {
        "type": "vggish_head",
        "output_index": 0,
        "post": "scale_0_1_to_1_10",
    },
    "gender-audioset-vggish": {
        "type": "vggish_head",
        "post": "gender_label",
    },
}

CAMELOT = [
    "1A", "2A", "3A", "4A", "5A", "6A", "7A", "8A",
    "9A", "10A", "11A", "12A",
    "1B", "2B", "3B", "4B", "5B", "6B", "7B", "8B",
    "9B", "10B", "11B", "12B",
]

MOOD_LABELS = [
    "Dark", "Melancholic", "Somber", "Neutral", "Warm", "Bright", "Uplifting"
]

VGGISH_MEL_CONFIG = {
    "sr": 16000,
    "n_mels": 96,
    "hop_length": 512,
    "n_fft": 2048,
    "fmin": 0,
    "fmax": 8000,
}


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class AIEngine:
    def __init__(self):
        self.sessions: dict[str, ort.InferenceSession] = {}
        self.vggish_session: ort.InferenceSession | None = None
        self.labels: dict[str, list[str]] = {}
        self.models_dir: str | None = None

    def load_model(self, slug: str, model_path: str, vggish_path: str | None = None):
        """Load an ONNX model into memory."""
        self.sessions[slug] = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

        # Load VGGish if needed and not already loaded
        config = MODEL_CONFIGS.get(slug, {})
        if config.get("type") == "vggish_head" and vggish_path and not self.vggish_session:
            self.vggish_session = ort.InferenceSession(
                vggish_path, providers=["CPUExecutionProvider"]
            )

        # Load labels if specified
        labels_file = config.get("labels_file")
        if labels_file and slug not in self.labels:
            labels_path = os.path.join(os.path.dirname(model_path), labels_file)
            if os.path.exists(labels_path):
                self.labels[slug] = json.load(open(labels_path))

    def unload_model(self, slug: str):
        """Unload a model from memory."""
        if slug in self.sessions:
            del self.sessions[slug]

    def run(self, slug: str, audio_path: str) -> str:
        """Run inference on an audio file. Returns formatted result string."""
        if slug not in self.sessions:
            raise ValueError(f"Model not loaded: {slug}")

        config = MODEL_CONFIGS.get(slug)
        if not config:
            raise ValueError(f"Unknown model: {slug}")

        model_type = config["type"]

        if model_type == "cqt_classification":
            return self._run_cqt(slug, audio_path, config)
        elif model_type == "mel_classification":
            return self._run_mel(slug, audio_path, config)
        elif model_type == "vggish_head":
            return self._run_vggish_head(slug, audio_path, config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _run_cqt(self, slug: str, audio_path: str, config: dict) -> str:
        """CQT-based models (key detection)."""
        wav, _ = librosa.load(audio_path, sr=config["sr"], mono=True, duration=config["duration"])
        cqt_cfg = config["cqt"]
        cqt = librosa.cqt(wav, sr=config["sr"], **cqt_cfg)
        spec = np.log1p(np.abs(cqt))[:, 0:-2]
        tensor = spec[np.newaxis, np.newaxis, :, :].astype(np.float32)

        session = self.sessions[slug]
        out = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        pred = int(np.argmax(out))

        return self._post_process(pred, out, config, slug)

    def _run_mel(self, slug: str, audio_path: str, config: dict) -> str:
        """Mel-spectrogram-based models (genre, BPM)."""
        wav, _ = librosa.load(audio_path, sr=config["sr"], mono=True, duration=config["duration"])
        mel_cfg = config["mel"]
        mel = librosa.feature.melspectrogram(y=wav, sr=config["sr"], **mel_cfg)

        # Apply log compression based on model requirements
        log_mode = config.get("log_compress", "log1p")
        if log_mode == "essentia":
            # Essentia TensorflowInputMusiCNN: log10(1 + mel * 10000) then z-normalize
            mel_db = np.log10(1.0 + mel * 10000.0)
            mel_db = (mel_db - 2.06755686098554) / 1.268292820667291
        else:
            mel_db = np.log1p(mel)

        fmt = config.get("tensor_format", "NCHW")
        if fmt == "NTM":
            mel_t = mel_db.T
            target = config.get("target_frames")
            if target:
                if mel_t.shape[0] < target:
                    mel_t = np.pad(mel_t, ((0, target - mel_t.shape[0]), (0, 0)))
                elif mel_t.shape[0] > target:
                    mel_t = mel_t[:target]
            tensor = mel_t[np.newaxis, :, :].astype(np.float32)
        elif fmt == "NHWC":
            tensor = mel_db[np.newaxis, :, :, np.newaxis].astype(np.float32)
        else:  # NCHW
            tensor = mel_db[np.newaxis, np.newaxis, :, :].astype(np.float32)

        session = self.sessions[slug]
        input_name = session.get_inputs()[0].name

        # Use specific output if configured
        output_name = config.get("output_name")
        if output_name:
            all_out = session.run(None, {input_name: tensor})
            out_names = [o.name for o in session.get_outputs()]
            idx = out_names.index(output_name) if output_name in out_names else 0
            out = all_out[idx]
        else:
            out = session.run(None, {input_name: tensor})[0]

        pred = int(np.argmax(out.flatten()))
        return self._post_process(pred, out, config, slug)

    def _run_vggish_head(self, slug: str, audio_path: str, config: dict) -> str:
        """VGGish feature extractor + classification head.
        Processes multiple 64-frame windows across the track and averages outputs."""
        if not self.vggish_session:
            raise ValueError("VGGish extractor not loaded")

        # Compute full mel-spectrogram for 30s of audio
        wav, _ = librosa.load(audio_path, sr=VGGISH_MEL_CONFIG["sr"], mono=True, duration=30)
        mel = librosa.feature.melspectrogram(y=wav, sr=VGGISH_MEL_CONFIG["sr"],
            n_mels=VGGISH_MEL_CONFIG["n_mels"],
            hop_length=VGGISH_MEL_CONFIG["hop_length"],
            n_fft=VGGISH_MEL_CONFIG["n_fft"],
            fmin=VGGISH_MEL_CONFIG["fmin"],
            fmax=VGGISH_MEL_CONFIG["fmax"])
        # Essentia log compression + z-normalization
        mel_db = np.log10(1.0 + mel * 10000.0)
        mel_db = (mel_db - 2.06755686098554) / 1.268292820667291
        mel_t = mel_db.T  # [frames, 96]

        # Split into non-overlapping 64-frame windows, process each through VGGish
        total_frames = mel_t.shape[0]
        window_size = 64
        all_outputs = []

        session = self.sessions[slug]
        input_name = session.get_inputs()[0].name

        for start in range(0, total_frames - window_size + 1, window_size):
            window = mel_t[start:start + window_size]
            vgg_tensor = window[np.newaxis, :, :].astype(np.float32)

            # VGGish embeddings for this window
            embeddings = self.vggish_session.run(None, {"melspectrogram": vgg_tensor})[0]

            # Classification head
            try:
                out = session.run(None, {input_name: embeddings})[0]
            except Exception:
                emb_3d = embeddings.reshape(1, 1, embeddings.shape[1])
                out = session.run(None, {input_name: emb_3d})[0]

            all_outputs.append(out.flatten())

        if not all_outputs:
            # Track too short — use whatever we have, padded
            window = mel_t[:window_size]
            if window.shape[0] < window_size:
                window = np.pad(window, ((0, window_size - window.shape[0]), (0, 0)))
            vgg_tensor = window[np.newaxis, :, :].astype(np.float32)
            embeddings = self.vggish_session.run(None, {"melspectrogram": vgg_tensor})[0]
            try:
                out = session.run(None, {input_name: embeddings})[0]
            except Exception:
                emb_3d = embeddings.reshape(1, 1, embeddings.shape[1])
                out = session.run(None, {input_name: emb_3d})[0]
            all_outputs.append(out.flatten())

        # Average outputs across all windows
        avg_out = np.mean(all_outputs, axis=0)

        return self._post_process(None, avg_out, config, slug)

    def _post_process(self, pred_idx, raw_out, config: dict, slug: str) -> str:
        """Convert raw model output to formatted string."""
        post = config.get("post", "")
        flat = raw_out.flatten()

        if post == "camelot_key":
            idx = (pred_idx % 12) + 1
            mode = "A" if pred_idx < 12 else "B"
            return f"{idx}{mode}"

        elif post == "bpm":
            return str(30 + int(np.argmax(flat)))

        elif post == "scale_1_9_to_1_10":
            oi = config.get("output_index", 0)
            val = float(flat[oi])
            scaled = ((val - 1) / (9 - 1)) * (10 - 1) + 1
            return str(round(max(1, min(10, scaled))))

        elif post == "scale_0_1_to_1_10":
            oi = config.get("output_index", 0)
            val = float(flat[oi])
            scaled = val * 9 + 1
            return str(round(max(1, min(10, scaled))))

        elif post == "mood_label":
            oi = config.get("output_index", 0)
            val = float(flat[oi])
            idx = round(((val - 1) / (9 - 1)) * 6)
            idx = max(0, min(len(MOOD_LABELS) - 1, idx))
            return MOOD_LABELS[idx]

        elif post == "gender_label":
            labels = ["female", "male"]
            return labels[int(np.argmax(flat))]

        else:
            # Classification with labels
            labels = self.labels.get(slug)
            if labels:
                idx = int(np.argmax(flat))
                if 0 <= idx < len(labels):
                    return labels[idx]
                return str(idx)
            return str(int(np.argmax(flat)))


# ---------------------------------------------------------------------------
# Server mode
# ---------------------------------------------------------------------------

def run_server(engine: AIEngine):
    """Read commands from stdin, write results to stdout."""
    # Suppress C-level stderr noise
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)

    # Warmup librosa imports
    _warmup = np.zeros(16000, dtype=np.float32)
    _ = librosa.feature.melspectrogram(y=_warmup, sr=16000, n_mels=96)

    print("READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        # Parse tab-separated: CMD\targ1\targ2\targ3
        # Falls back to space-split for simple commands (QUIT, READY)
        if "\t" in line:
            parts = line.split("\t")
        else:
            parts = line.split(" ", 1)

        cmd = parts[0].upper()

        if cmd == "QUIT":
            break

        elif cmd == "LOAD":
            # LOAD\tslug\tmodel_path[\tvggish_path]
            tab_parts = line.split("\t")
            if len(tab_parts) < 3:
                print(f"ERROR load LOAD requires: LOAD\\tslug\\tmodel_path", flush=True)
                continue
            slug = tab_parts[1]
            model_path = tab_parts[2]
            vggish_path = tab_parts[3] if len(tab_parts) > 3 else None
            try:
                engine.load_model(slug, model_path, vggish_path)
                print(f"LOADED {slug}", flush=True)
            except Exception as e:
                print(f"ERROR {slug} {e}", flush=True)

        elif cmd == "RUN":
            # RUN\tslug\taudio_path
            tab_parts = line.split("\t")
            if len(tab_parts) < 3:
                print(f"ERROR run RUN requires: RUN\\tslug\\taudio_path", flush=True)
                continue
            slug = tab_parts[1]
            audio_path = tab_parts[2]
            try:
                result = engine.run(slug, audio_path)
                print(f"RESULT {slug} {result}", flush=True)
            except Exception as e:
                print(f"ERROR {slug} {e}", flush=True)

        elif cmd == "UNLOAD":
            tab_parts = line.split("\t")
            slug = tab_parts[1] if len(tab_parts) > 1 else ""
            engine.unload_model(slug)
            print(f"UNLOADED {slug}", flush=True)

        else:
            print(f"ERROR unknown Unknown command: {line.strip()}", flush=True)


# ---------------------------------------------------------------------------
# Single-file mode
# ---------------------------------------------------------------------------

def run_single(engine: AIEngine, slug: str, model_path: str, audio_path: str,
               vggish_path: str | None = None):
    """Run a single inference and print the result."""
    engine.load_model(slug, model_path, vggish_path)
    result = engine.run(slug, audio_path)
    print(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    engine = AIEngine()

    if "--server" in sys.argv:
        run_server(engine)
    elif "--run" in sys.argv:
        idx = sys.argv.index("--run")
        slug = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        model_path = None
        audio_path = None
        vggish_path = None

        if "--model-path" in sys.argv:
            i = sys.argv.index("--model-path")
            model_path = sys.argv[i + 1]
        if "--file" in sys.argv:
            i = sys.argv.index("--file")
            audio_path = sys.argv[i + 1]
        if "--vggish-path" in sys.argv:
            i = sys.argv.index("--vggish-path")
            vggish_path = sys.argv[i + 1]

        if not all([slug, model_path, audio_path]):
            print("Usage: spintools-ai --run <slug> --model-path <path> --file <path>", file=sys.stderr)
            sys.exit(1)

        run_single(engine, slug, model_path, audio_path, vggish_path)
    else:
        print("Usage:", file=sys.stderr)
        print("  spintools-ai --server", file=sys.stderr)
        print("  spintools-ai --run <slug> --model-path <path> --file <path>", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
