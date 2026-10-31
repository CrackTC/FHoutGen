"""fhout for voices: one JSON in the game's shape for a set of wav files, each file's name (without .wav) as its TextId.

    .venv/Scripts/python -I python/generate.py runs/v1/model.pt out.json voice1.wav voice2.wav ...
    .venv/Scripts/python -I python/generate.py runs/v1/model.pt out.json <dir of wavs>

The voices may be at any sample rate, mono or stereo (16/32-bit PCM or 8-bit); they are resampled to 16 kHz. The values
are rounded to 3 decimals, as the official files are.
"""
import datetime
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as fx  # noqa: E402
from train import load_model, predict  # noqa: E402


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    model_path, out_path, inputs = sys.argv[1], sys.argv[2], sys.argv[3:]
    files = []
    for item in inputs:
        if os.path.isdir(item):
            files += sorted(os.path.join(item, f) for f in os.listdir(item) if f.lower().endswith('.wav'))
        else:
            files.append(item)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(os.path.dirname(model_path), device)
    body = []
    for path in files:
        pcm, rate = fx.read_wav(path)
        values = predict(model, fx.features(pcm, rate), device)
        body.append({
            'TextId': os.path.splitext(os.path.basename(path))[0],
            'List': [{'t': i, 'v': round(float(v), 3)} for i, v in enumerate(values)],
        })
    header = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'Header': header, 'Body': body}, f, ensure_ascii=False, separators=(',', ':'))
    print(f'{len(body)} voices -> {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
