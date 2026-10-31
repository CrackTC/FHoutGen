"""Features for the fhout model: a log-mel frame per 1/60 s of a voice, aligned with the fhout frames.

Frame i covers the voice from i/60 s to (i+1)/60 s; its window (WINDOW samples at RATE) is centred on that span's middle.
The same function serves the training cache (`python -I python/features.py data/pairs data/features`) and inference.
"""
import json
import os
import sys
import wave

import numpy as np

RATE = 16000
FPS = 60
WINDOW = 512          # 32 ms
N_MELS = 64
FMIN, FMAX = 60.0, 7600.0


def read_wav(path):
    with wave.open(path, 'rb') as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width == 2:
        data = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype='<i4').astype(np.float32) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f'{path}: {width * 8}-bit samples')
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def resample(pcm, rate, to=RATE):
    if rate == to:
        return pcm
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(rate, to)
    return resample_poly(pcm, to // g, rate // g).astype(np.float32)


def _mel_filters():
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    bins = WINDOW // 2 + 1
    freqs = np.linspace(0, RATE / 2, bins)
    mels = np.linspace(hz_to_mel(FMIN), hz_to_mel(FMAX), N_MELS + 2)
    hz = mel_to_hz(mels)
    fb = np.zeros((N_MELS, bins), dtype=np.float32)
    for m in range(N_MELS):
        lo, mid, hi = hz[m], hz[m + 1], hz[m + 2]
        up = (freqs - lo) / (mid - lo)
        down = (hi - freqs) / (hi - mid)
        fb[m] = np.maximum(0, np.minimum(up, down))
    return fb


_FB = _mel_filters()
_HANN = np.hanning(WINDOW).astype(np.float32)


def frame_count(samples, rate):
    """The fhout length of a voice: a frame per 1/60 s, rounded up."""
    return int(np.ceil(samples * FPS / rate))


def features(pcm, rate, frames=None):
    """(frames, N_MELS + 1) float32: log-mel (natural log of power) and the frame's log energy."""
    if frames is None:
        frames = frame_count(len(pcm), rate)
    pcm = resample(pcm, rate)
    hop = RATE / FPS
    centres = np.round((np.arange(frames) + 0.5) * hop).astype(np.int64)
    padded = np.concatenate([np.zeros(WINDOW, np.float32), pcm, np.zeros(WINDOW, np.float32)])
    idx = centres[:, None] - WINDOW // 2 + WINDOW + np.arange(WINDOW)[None, :]
    idx = np.clip(idx, 0, len(padded) - 1)
    segs = padded[idx] * _HANN
    power = np.abs(np.fft.rfft(segs, axis=1)) ** 2
    mel = np.log(power @ _FB.T + 1e-10)
    energy = np.log(np.mean(segs ** 2, axis=1) + 1e-10)
    return np.concatenate([mel, energy[:, None]], axis=1).astype(np.float32)


def build_cache(root, out):
    """data/features/{feats.npy (all frames, float16), targets.npy (v, float16), index.json ([sheet, cue, start, n])}."""
    os.makedirs(out, exist_ok=True)
    sheets = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    feats, targets, index = [], [], []
    start = 0
    for k, sheet in enumerate(sheets):
        d = os.path.join(root, sheet)
        try:
            body = json.load(open(os.path.join(d, 'fhout.json'), encoding='utf-8-sig'))['Body']
        except (OSError, ValueError):
            continue
        for rec in body:
            path = os.path.join(d, rec['TextId'] + '.wav')
            if not os.path.exists(path):
                continue
            v = np.array([p['v'] for p in rec['List']], dtype=np.float32)
            pcm, rate = read_wav(path)
            f = features(pcm, rate, len(v))
            feats.append(f.astype(np.float16))
            targets.append(v.astype(np.float16))
            index.append([sheet, rec['TextId'], start, len(v)])
            start += len(v)
        if (k + 1) % 200 == 0:
            print(f'{k + 1}/{len(sheets)} sheets, {start / FPS / 3600:.1f} h', flush=True)
    np.save(os.path.join(out, 'feats.npy'), np.concatenate(feats))
    np.save(os.path.join(out, 'targets.npy'), np.concatenate(targets))
    json.dump(index, open(os.path.join(out, 'index.json'), 'w'))
    print(f'{len(index)} cues, {start} frames ({start / FPS / 3600:.1f} h)')


if __name__ == '__main__':
    build_cache(sys.argv[1], sys.argv[2])
