"""Where the model's error is, on the validation sheets: by frame state, near edges, in the official slow ramps, by kind.

    .venv/Scripts/python -I python/analyze.py data/features runs/v1/model.pt
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import load_model, predict, is_val  # noqa: E402


def main():
    cache, model_path = sys.argv[1], sys.argv[2]
    feats = np.load(os.path.join(cache, 'feats.npy'), mmap_mode='r')
    targets = np.load(os.path.join(cache, 'targets.npy')).astype(np.float32)
    index = [e for e in json.load(open(os.path.join(cache, 'index.json'))) if is_val(e[0])]
    device = 'cuda'
    model = load_model(os.path.dirname(model_path), device)

    cats = defaultdict(lambda: [0, 0.0])
    kinds = defaultdict(lambda: [0, 0.0])
    ramp_runs = []
    total_err = 0.0
    for sheet, cue, s, n in index:
        t = targets[s:s + n]
        p = predict(model, feats[s:s + n].astype(np.float32), device)
        e = np.abs(p - t)
        total_err += e.sum()
        kinds['_'.join(sheet.split('_')[:2])][0] += n
        kinds['_'.join(sheet.split('_')[:2])][1] += e.sum()
        on_t, on_p = t > 0, p > 0
        d = np.diff(t, prepend=t[0])
        # official linear ramps: >= 6 frames of a constant negative or positive step
        ramp = np.zeros(n, bool)
        i = 1
        while i < n:
            if abs(d[i]) > 1e-4:
                j = i
                while j + 1 < n and abs(d[j + 1] - d[i]) < 2e-3:
                    j += 1
                if j - i + 1 >= 6:
                    ramp[i:j + 1] = True
                    ramp_runs.append(j - i + 1)
                i = j + 1
            else:
                i += 1
        edge = np.zeros(n, bool)
        for k in np.flatnonzero(np.diff(on_t.astype(int)) != 0):
            edge[max(0, k - 2):k + 3] = True
        for name, mask in (
                ('both closed', ~on_t & ~on_p), ('false open', ~on_t & on_p), ('missed open', on_t & ~on_p),
                ('both open', on_t & on_p), ('within 2 frames of an official edge', edge),
                ('official slow ramp (>=6 frames)', ramp)):
            cats[name][0] += mask.sum()
            cats[name][1] += e[mask].sum()
    frames = sum(e[3] for e in index)
    print(f'{frames} frames, MAE {total_err / frames:.4f}')
    print(f'{"frames of":40} {"share":>6} {"MAE":>6} {"share of error":>14}')
    for name, (n, err) in cats.items():
        print(f'{name:40} {n / frames:6.3f} {err / max(n, 1):6.3f} {err / total_err:14.3f}')
    if ramp_runs:
        print(f'slow ramps: {len(ramp_runs)}, length median {np.median(ramp_runs):.0f}, p90 {np.percentile(ramp_runs, 90):.0f}')
    print('by kind (top 10 by frames):')
    for k, (n, err) in sorted(kinds.items(), key=lambda x: -x[1][0])[:10]:
        print(f'  {k:28} {n / 60 / 3600:5.2f} h  MAE {err / n:.4f}')


if __name__ == '__main__':
    main()
