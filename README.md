# FHoutGen

Generates the game's fhout lip-sync data (the 60 Hz mouth-open series `{Header, Body: [{TextId, List: [{t, v}]}]}` that
the demo reads from `fhout_<sheet>.json.unity3d`) from a voice alone, for voices that have none (AI-generated voices).

## Use

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install numpy scipy
.venv/Scripts/python -I python/generate.py models/full128/model.pt out.json voice1.wav voice2.wav   # or a folder of wavs
```

Each wav (any rate, mono or stereo) becomes one record, its file name as the TextId. The voice must be the voice alone:
background music makes the mouth stay open (separate the vocals first).

`models/full128`: hidden 128 (0.45 M parameters, 1.8 MB), trained on all 240 h of local official fhout. On the held-out
sheets (22 h): MAE 0.033, open/closed agreement 99.3 %, correlation 0.93 (the loudness rule of `python/baseline.py`:
0.096, 95.5 %, 0.71). About 78x real time on the CPU.

## Train

- `Extract/` (.NET 10): `fhout-extract stats|export` reads the demo's local `fhout_*.json.unity3d` bundles and decodes
  each record's voice cue from `<sheet>.acb.unity3d` with the demo's vendored DereTore source, writing pairs
  `<out>/<sheet>/{<cue>.wav, fhout.json}` (mono 16-bit, 16 kHz by default).
- `python/features.py`: the 60 Hz log-mel frames (and the cache of a pair folder); `build_full.sh` caches every sheet a
  sixth at a time and merges them with `python/merge_features.py`.
- `python/train.py <cache> <run>`: training and validation (every 10th sheet by name hash is held out);
  `python/analyze.py` splits a model's validation error.

```bash
dotnet build Extract -c Release
bash build_full.sh
.venv/Scripts/python -I python/train.py data/features_full runs/full128 --epochs 20
```
