"""
Key detection using MusicalKeyCNN.
Uses librosa for CQT computation and onnxruntime for ONNX model inference.

Modes:
    Single file:  detect_key /path/to/track.mp3
    Server mode:  detect_key --server
                  (reads file paths from stdin, writes keys to stdout, one per line)
"""

import sys
import os
import warnings
import numpy as np

# Suppress librosa/soundfile warnings about corrupt MP3 frames
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import librosa
import onnxruntime as ort


CAMELOT = [
    "1A", "2A", "3A", "4A", "5A", "6A", "7A", "8A",
    "9A", "10A", "11A", "12A",
    "1B", "2B", "3B", "4B", "5B", "6B", "7B", "8B",
    "9B", "10B", "11B", "12B",
]

SAMPLE_RATE = 44100
N_BINS = 105
BINS_PER_OCTAVE = 24
FMIN = 65
HOP_LENGTH = 8820


def get_bundled_model_path():
    """Find the keynet.onnx model bundled with PyInstaller or in dev."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, 'keynet.onnx')
    return os.path.join(os.path.dirname(__file__), '..', 'models', 'keynet.onnx')


def detect_key(audio_path: str, session: ort.InferenceSession) -> str:
    waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    cqt = librosa.cqt(
        waveform,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE,
        fmin=FMIN,
    )
    spec = np.abs(cqt)
    spec = np.log1p(spec)
    spec = spec[:, 0:-2]

    input_tensor = spec[np.newaxis, np.newaxis, :, :].astype(np.float32)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    pred = int(np.argmax(outputs[0], axis=1)[0])

    idx = (pred % 12) + 1
    mode = "A" if pred < 12 else "B"
    return f"{idx}{mode}"


def run_server(session: ort.InferenceSession):
    """Read file paths from stdin, output keys to stdout. One per line."""
    # Suppress C-level stderr noise (mpg123 decoder warnings)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)

    # Warmup: force all lazy imports by processing a silent buffer
    _warmup = np.zeros(SAMPLE_RATE, dtype=np.float32)
    _ = librosa.cqt(_warmup, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                     n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, fmin=FMIN)

    # Signal ready (all imports loaded, model warm)
    print("READY", flush=True)

    for line in sys.stdin:
        audio_path = line.strip()
        if not audio_path:
            continue
        if audio_path == "QUIT":
            break

        try:
            key = detect_key(audio_path, session)
            print(key, flush=True)
        except Exception as e:
            print(f"ERROR:{e}", flush=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: detect_key <audio_file>", file=sys.stderr)
        print("       detect_key --server", file=sys.stderr)
        sys.exit(1)

    model_path = get_bundled_model_path()
    if not os.path.exists(model_path):
        # Check if passed as second arg
        if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
            model_path = sys.argv[2]
        else:
            print(f"Model not found: {model_path}", file=sys.stderr)
            sys.exit(1)

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    if sys.argv[1] == "--server":
        run_server(session)
    else:
        audio_path = sys.argv[1]
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}", file=sys.stderr)
            sys.exit(1)
        key = detect_key(audio_path, session)
        print(key)


if __name__ == "__main__":
    main()
