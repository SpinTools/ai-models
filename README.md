# SpinTools AI Models

Distribution point for the AI analysis models and engine used by
[SpinTools](https://spintools.io).

The SpinTools desktop app reads `manifest.json` from this repository to see
which models are available, then fetches the ones you turn on from this
repository's releases. Nothing here needs to be downloaded by hand.

## What's here

- **`manifest.json`** — the catalog the app reads: model names, descriptions,
  versions, and where to download each one.
- **Releases** — the model files (`.onnx`) and the analysis engine binaries for
  macOS, Windows, and Linux.

## Models

| Model | What it does |
| --- | --- |
| Genre Tagger | Tags a track with one of 500+ Discogs styles |
| Key Finder | Detects musical key so you can mix in key |
| BPM Detector | Detects tempo, including on syncopated rhythms |
| Energy Level | Scores a track from chill warm-up to peak-time |
| Mood | Scores a track from dark and moody to bright and uplifting |
| Danceability | Rates how danceable a track is |
| Vocal Type | Detects feminine, masculine, or mixed vocal character |

## Licensing

The model files distributed here are third-party artifacts and remain subject
to their original licenses.
