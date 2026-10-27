# Model licences and attribution

The model files distributed from this repository are third-party artifacts
and remain subject to their original licences. This file records the licence
for each redistributed file and carries the attribution those licences
require.

Every mirrored file, except the fine-tuned beat tracker described under
"Beat tracking model", is **byte-identical to its upstream original**. The
`sha256` below is the upstream file's own digest, and it is verified on
every mirror run — so anyone can confirm that what is served here is exactly
what the original author published.

## Separation models

### `vocals_mel_band_roformer.ckpt`

| | |
| --- | --- |
| **Author** | Kimberley Jensen |
| **Licence** | MIT |
| **Upstream** | <https://huggingface.co/KimberleyJSN/melbandroformer> (`MelBandRoformer.ckpt`) |
| **Project** | <https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model> |
| **sha256** | `87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e` |
| **Size** | 913,106,900 bytes |

The MIT grant is declared by the rightsholder: the author's own project
repository links to exactly this file as the model download. The full
licence text and attribution ship beside the weights as
`vocals_mel_band_roformer.LICENSE.txt` on the same release, so the notice
travels with the bytes.

The file is published here under a different name from upstream. Only the
name differs — the digest above is the upstream file's. The rename is
functional: the separation backend selects a model architecture by matching
the installed filename against its own registry.

The Mel-Band Roformer *architecture* is separate work again: the reference
implementation (<https://github.com/lucidrains/BS-RoFormer>, Phil Wang) is
MIT and ships no weights. An architecture's licence does not carry over to a
checkpoint trained on it.

## Beat tracking model

### `beatthis-final-ftd-serato-1.onnx`

| | |
| --- | --- |
| **Authors** | Francesco Foscarin, Jan Schluter and Gerhard Widmer (Institute of Computational Perception, Johannes Kepler University Linz) |
| **Licence** | MIT |
| **Upstream** | <https://github.com/CPJKU/beat_this> (the `final0` checkpoint) |
| **Paper** | "Beat This! Accurate beat tracking without DBN postprocessing", ISMIR 2024 |
| **sha256** | `3f0cd8747fa83542437e4f40e9ea47fc6f319053b3069433ec9695f588754167` |
| **Size** | 82,568,067 bytes |

Unlike the files above, this one is **not** byte-identical to the upstream
checkpoint. It is the upstream model fine-tuned on beat targets for DJ
beatgrid alignment and exported to ONNX (static `1 x 1500 x 128` log-mel
input, opset 17). The upstream code and published weights are MIT-licensed
and this derivative is redistributed under the same terms. The fine-tuning
data is not redistributed. The full licence text and attribution ship beside
the weights as `BEATTHIS-NOTICE.txt` on the `weights-v2` release.

## Analysis models

The `.onnx` analysis models on the `v1` release are third-party artifacts
from their respective upstreams and remain under their original licences.
Consult each model's upstream project for its terms.

## Engine binaries

The `spintools-ai-*` binaries are SpinTools' own analysis engine. They embed
third-party open-source Python libraries, each under its own licence.

### Native libraries in the macOS arm64 engine (engine-v9 onward)

The macOS arm64 engine's audio analysis uses Essentia, built from source by
SpinTools with the libraries below linked statically into one extension module.
Each is built from the unmodified upstream source named in
[`ENGINE_NOTICES.txt`](ENGINE_NOTICES.txt), which also quotes each licence
verbatim from that source.

| Library | Version | Licence |
| --- | --- | --- |
| Essentia | 2.1-beta6-dev (MTG/essentia `86bdb8e3`) | AGPL-3.0-only |
| FFmpeg | 7.1.1 (decode-only, no GPL or non-free components) | LGPL-2.1-or-later |
| TagLib | 1.13.1 | LGPL-2.1 or MPL-1.1 |
| libsamplerate | 0.2.2 | BSD-2-Clause |
| libyaml | 0.2.5 | MIT |
| Chromaprint | 1.5.1 | LGPL-2.1 (as a whole, per its own `LICENSE.md`; its own code is MIT, with FFmpeg parts under LGPL-2.1) |
| FFTW | 3.3.10 | GPL-2.0-or-later |
| Eigen | 3.3.7 (headers) | MPL-2.0 |

The previous macOS engine (engine-v8) carried Essentia as a PyPI wheel bundling
41 libraries, including `libx264` and `libx265` (GPL-2.0-or-later) and LAME.
**None of those are in engine-v9.** That is not a claim that the engine is
LGPL-only: it contains FFTW (GPL-2.0-or-later) and Essentia (AGPL-3.0), as
engine-v8 did through the same wheel. The Windows and Linux engines are
unchanged from engine-v4 and are not covered by this table.

They also carry one third-party **data** file, bundled rather than
downloaded so that a separation never has to reach a host outside this
repository:

### `vocals_mel_band_roformer.yaml`

| | |
| --- | --- |
| **Author** | the `python-audio-separator` project |
| **Licence** | MIT |
| **Upstream** | <https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/vocals_mel_band_roformer.yaml> |
| **Project** | <https://github.com/nomadkaraoke/python-audio-separator> |
| **sha256** | `b958b29c8f7195f0d86bee6759a33980db675c4ecaf2fcaa80fa125828e6cd38` |
| **Size** | 944 bytes |

The separation backend refuses to load the checkpoint without this config.
It is carried inside the engine, byte-identical to the file upstream serves.

**It is not the checkpoint author's own config**, and the distinction
matters: Kimberley Jensen's repository ships
`configs/config_vocals_mel_band_roformer.yaml`, and the file above is that
config adapted for this backend — the same architecture block, plus the
`audio:` block this loader requires. Redistribution therefore rests on
`python-audio-separator`'s MIT grant, not on the checkpoint's.
