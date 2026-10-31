"""Rule baseline for fhout: the 60 Hz mouth-open series from a voice's loudness alone.

    python -I python/baseline.py data/pairs [--max-sheets N] [--threshold -60] [--hold 0.42]

Per frame (1/60 s, as the game's fhout): the RMS of the voice in dB. A frame is open when the level is above the
threshold (with a hysteresis of a few dB and short gaps bridged); an open frame takes the hold value, and the opening and
closing ramp over two frames, as the official data does. Prints how close that comes to the official fhout, overall and
per kind of voice (the sheet name's first two parts).
"""
import argparse
import json
import os
import sys
import wave
from collections import defaultdict

import numpy as np

FPS = 60


def read_wav(path):
    with wave.open(path, 'rb') as w:
        rate = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float32) / 32768.0
    return data, rate


def frame_db(pcm, rate, frames):
    hop = rate / FPS
    out = np.full(frames, -120.0, dtype=np.float32)
    for i in range(frames):
        seg = pcm[int(round(i * hop)):int(round((i + 1) * hop))]
        if len(seg):
            out[i] = 10 * np.log10(np.mean(seg.astype(np.float64) ** 2) + 1e-12)
    return out


def rule(db, threshold, hysteresis, bridge, hold):
    n = len(db)
    on = np.zeros(n, dtype=bool)
    state = False
    for i in range(n):
        state = db[i] > (threshold - hysteresis if state else threshold)
        on[i] = state
    # bridge short closed gaps between open runs
    i = 0
    while i < n:
        if not on[i]:
            j = i
            while j < n and not on[j]:
                j += 1
            if 0 < i and j < n and j - i <= bridge:
                on[i:j] = True
            i = j
        else:
            i += 1
    v = np.where(on, hold, 0.0).astype(np.float32)
    # two-frame ramps at the edges (0 -> half -> hold, and back)
    out = v.copy()
    for i in range(n):
        if on[i] and (i == 0 or not on[i - 1]):
            out[i] = hold / 2
        if on[i] and (i == n - 1 or not on[i + 1]):
            out[i] = hold / 2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--max-sheets', type=int, default=0)
    ap.add_argument('--threshold', type=float, default=-60.0)
    ap.add_argument('--hysteresis', type=float, default=6.0)
    ap.add_argument('--bridge', type=int, default=3)
    ap.add_argument('--hold', type=float, default=0.42)
    args = ap.parse_args()

    sheets = sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))
    if args.max_sheets:
        step = max(1, len(sheets) // args.max_sheets)
        sheets = sheets[::step][:args.max_sheets]
    totals = defaultdict(lambda: np.zeros(5))  # frames, abs err, open agreement, sum v*p, count for corr later
    series = defaultdict(lambda: ([], []))
    cues = 0
    for sheet in sheets:
        d = os.path.join(args.root, sheet)
        try:
            body = json.load(open(os.path.join(d, 'fhout.json'), encoding='utf-8-sig'))['Body']
        except (OSError, ValueError):
            continue
        kind = '_'.join(sheet.split('_')[:2])
        for rec in body:
            path = os.path.join(d, rec['TextId'] + '.wav')
            if not os.path.exists(path):
                continue
            truth = np.array([p['v'] for p in rec['List']], dtype=np.float32)
            pcm, rate = read_wav(path)
            pred = rule(frame_db(pcm, rate, len(truth)), args.threshold, args.hysteresis, args.bridge, args.hold)
            for key in (kind, '*all*'):
                t = totals[key]
                t[0] += len(truth)
                t[1] += np.abs(pred - truth).sum()
                t[2] += ((pred > 0) == (truth > 0)).sum()
                series[key][0].append(truth)
                series[key][1].append(pred)
            cues += 1
    print(f'{cues} cues from {len(sheets)} sheets; threshold {args.threshold} dB, hysteresis {args.hysteresis}, '
          f'bridge {args.bridge}, hold {args.hold}')
    print(f'{"kind":28} {"hours":>6} {"MAE":>6} {"open agr":>8} {"corr":>6}')
    for key in sorted(totals, key=lambda k: (k != '*all*', -totals[k][0])):
        t = totals[key]
        truth = np.concatenate(series[key][0])
        pred = np.concatenate(series[key][1])
        corr = np.corrcoef(truth, pred)[0, 1] if truth.std() > 0 and pred.std() > 0 else float('nan')
        print(f'{key:28} {t[0] / FPS / 3600:6.2f} {t[1] / t[0]:6.3f} {t[2] / t[0]:8.3f} {corr:6.3f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
