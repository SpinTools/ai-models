# Model licences and attribution

The model files distributed from this repository are third-party artifacts
and remain subject to their original licences. This file records the licence
for each redistributed file and carries the attribution those licences
require.

Every mirrored file is **byte-identical to its upstream original**. The
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

## Analysis models

The `.onnx` analysis models on the `v1` release are third-party artifacts
from their respective upstreams and remain under their original licences.
Consult each model's upstream project for its terms.

## Engine binaries

The `spintools-ai-*` binaries are SpinTools' own analysis engine. They embed
third-party open-source Python libraries, each under its own licence.

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
