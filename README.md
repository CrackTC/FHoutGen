# FHoutGen

Generates the game's fhout lip-sync data (the 60 Hz mouth-open series `{Header, Body: [{TextId, List: [{t, v}]}]}` that
the demo reads from `fhout_<sheet>.json.unity3d`) from a voice alone, for voices that have none (AI-generated voices).

## Use

### Windows executable (no Python)

```bash
dotnet publish Generate -c Release -r win-x64 -o publish
publish/fhout-gen.exe -o out.json voice1.wav voice2.mp3 <folder> ...
```

`fhout-gen.exe` is one self-contained file (about 45 MB, the model embedded; `--model x.onnx` takes another). It reads
wav and mp3 (any rate, mono or stereo; a folder gives its .wav and .mp3 files in name order) and writes the same JSON as
`python/generate.py`; without `-o` the JSON goes to standard output. Its features are the Python ones step for step
(scipy's `resample_poly`, numpy's Hann window and percentile): on the same PCM the two agree to the third decimal. An mp3
is decoded by NLayer (managed) behind NAudio's frame reader, as ffmpeg decodes it (Media Foundation would start 528
samples late; NAudio's ACM decoder rounds differently and does not run under native AOT).

Native AOT (needs the MSVC build tools; `vswhere.exe` must be on PATH):

```bash
PATH="/c/Program Files (x86)/Microsoft Visual Studio/Installer:$PATH" \
  dotnet publish Generate -c Release -r win-x64 -p:PublishAot=true -o publish-aot
```

gives a 4.5 MB `fhout-gen.exe` that has to stay beside `onnxruntime.dll` and `onnxruntime_providers_shared.dll` (native
DLLs cannot go inside a native AOT exe). Same output; 18 mp3 voices (95 s) in about 1.1 s against 1.6 s for the
single-file build, mostly the start-up.

`models/full128/model.onnx` comes from `python/export_onnx.py models/full128 models/full128/model.onnx`.

### Python

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
