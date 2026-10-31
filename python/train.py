"""Trains the fhout model on the feature cache and scores it, with the rule baseline, on held-out sheets.

    .venv/Scripts/python -I python/train.py data/features runs/v1 [--epochs 30]

Model: per-cue normalised log-mel frames -> 3 dilated 1-D convolutions -> a bidirectional GRU -> per frame an "open"
logit, a mouth value and (with --levels) a choice among the official tool's fixed levels or "free". The output is 0 where
closed, the chosen level where a level is chosen, else the value. Loss: BCE on open, cross-entropy on the level choice
(open frames), L1 on the value (open frames that are on no level, or all open frames without levels). The input
level is shifted at random each batch (a gain of +-12 dB is a constant in log power), so the model does not lean on the
absolute loudness of the official recordings. Validation: every 10th sheet (by name hash), never trained on.
"""
import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as fx  # noqa: E402

N_IN = fx.N_MELS + 1
CHUNK = 600  # frames per training sequence (10 s)
# The official fhout's most frequent exact values (a third of the open frames sit on one of them; 0.42 alone is 21 %).
LEVELS = [0.42, 0.005, 0.30, 0.36, 0.25, 0.385]
LEVEL_TOLERANCE = 6e-4


class FHoutNet(nn.Module):
    def __init__(self, hidden=128, levels=()):
        super().__init__()
        self.levels = list(levels)
        self.inp = nn.Conv1d(N_IN, hidden, 1)
        self.convs = nn.ModuleList(nn.Conv1d(hidden, hidden, 5, padding=2 * d, dilation=d) for d in (1, 2, 4))
        self.norms = nn.ModuleList(nn.GroupNorm(1, hidden) for _ in range(3))
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, 2 + (len(self.levels) + 1 if self.levels else 0))

    def forward(self, x):  # x: (B, T, N_IN), already normalised
        h = self.inp(x.transpose(1, 2))
        for conv, norm in zip(self.convs, self.norms):
            h = h + F.gelu(norm(conv(h)))
        h, _ = self.gru(h.transpose(1, 2))
        o = self.out(h)
        # open logit, value in (0, 1), level logits (the last class is "free") or None
        return o[..., 0], torch.sigmoid(o[..., 1]), (o[..., 2:] if self.levels else None)


def normalise(f):
    """Per cue: the log-mel and energy relative to the cue's loudest frames (its 95th percentile energy)."""
    ref = np.percentile(f[:, -1], 95)
    return (f - ref) / 5.0


def predict(model, feats, device):
    x = torch.from_numpy(normalise(feats)[None]).to(device)
    with torch.no_grad():
        logit, value, level_logits = model(x)
    if level_logits is not None:
        table = torch.tensor(model.levels, device=value.device)
        choice = level_logits.argmax(-1)
        free = choice == len(model.levels)
        value = torch.where(free, value, table[choice.clamp(max=len(model.levels) - 1)])
    return torch.where(logit > 0, value, torch.zeros_like(value))[0].cpu().numpy()


def level_targets(y):
    """Per frame the index of the fixed level y sits on, else len(LEVELS) ("free")."""
    cls = torch.full(y.shape, len(LEVELS), dtype=torch.long, device=y.device)
    for i, level in enumerate(LEVELS):
        cls[(y - level).abs() < LEVEL_TOLERANCE] = i
    return cls


def load_model(run_dir, device):
    """The run's model: model.pt with config.json (hidden, levels); runs before config.json are hidden 128, no levels."""
    config_path = os.path.join(run_dir, 'config.json')
    config = json.load(open(config_path)) if os.path.exists(config_path) else {'hidden': 128, 'levels': []}
    model = FHoutNet(config['hidden'], config['levels']).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, 'model.pt'), map_location=device))
    model.eval()
    return model


def is_val(sheet):
    return zlib.crc32(sheet.encode()) % 10 == 0


def baseline(feats):
    """The rule of python/baseline.py on the cached energy (log power): -60 dB, 6 dB hysteresis, 3-frame bridge."""
    db = feats[:, -1].astype(np.float64) * (10 / np.log(10))
    n = len(db)
    on = np.zeros(n, bool)
    state = False
    for i in range(n):
        state = db[i] > (-66 if state else -60)
        on[i] = state
    i = 0
    while i < n:
        if not on[i]:
            j = i
            while j < n and not on[j]:
                j += 1
            if 0 < i and j < n and j - i <= 3:
                on[i:j] = True
            i = j
        else:
            i += 1
    out = np.where(on, 0.42, 0.0)
    for i in range(n):
        if on[i] and (i == 0 or not on[i - 1] or i == n - 1 or not on[i + 1]):
            out[i] = 0.21
    return out.astype(np.float32)


def score(pairs):
    truth = np.concatenate([t for t, _ in pairs])
    pred = np.concatenate([p for _, p in pairs])
    return {
        'mae': float(np.abs(pred - truth).mean()),
        'open': float(((pred > 0) == (truth > 0)).mean()),
        'corr': float(np.corrcoef(pred, truth)[0, 1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cache')
    ap.add_argument('out')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--train-fraction', type=float, default=1.0)
    ap.add_argument('--levels', action='store_true', help='the fixed-level head')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    feats = np.load(os.path.join(args.cache, 'feats.npy'), mmap_mode='r')
    targets = np.load(os.path.join(args.cache, 'targets.npy')).astype(np.float32)
    index = json.load(open(os.path.join(args.cache, 'index.json')))
    train = [e for e in index if not is_val(e[0])]
    if args.train_fraction < 1:
        sheets = sorted({e[0] for e in train})
        keep = set(sheets[::max(1, round(1 / args.train_fraction))])
        train = [e for e in train if e[0] in keep]
    val = [e for e in index if is_val(e[0])]
    print(f'train {len(train)} cues, val {len(val)} cues ({sum(e[3] for e in val) / 3600 / 60:.1f} h)', flush=True)

    # Training sequences: each cue cut into CHUNK-frame pieces (the last one padded and masked).
    pieces = [(s, min(CHUNK, n - o)) for _, _, s0, n in train for o in range(0, n, CHUNK) for s in [s0 + o]]
    levels = LEVELS if args.levels else []
    model = FHoutNet(args.hidden, levels).to(device)
    json.dump({'hidden': args.hidden, 'levels': levels}, open(os.path.join(args.out, 'config.json'), 'w'))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (len(pieces) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    rng = np.random.default_rng(0)
    # per-cue normalisation reference for each piece's cue
    cue_ref = {}
    for _, _, s0, n in train:
        cue_ref[s0] = np.percentile(feats[s0:s0 + n, -1].astype(np.float32), 95)
    piece_ref = np.array([cue_ref[s0] for _, _, s0, n in train for o in range(0, n, CHUNK)], dtype=np.float32)

    best = None
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(pieces))
        t0, total = time.time(), 0.0
        for b in range(len(pieces) // args.batch):
            ids = order[b * args.batch:(b + 1) * args.batch]
            x = np.zeros((len(ids), CHUNK, N_IN), np.float32)
            y = np.zeros((len(ids), CHUNK), np.float32)
            m = np.zeros((len(ids), CHUNK), np.float32)
            for k, i in enumerate(ids):
                s, n = pieces[i]
                # gain augmentation: shift the level of the input only (the reference stays the cue's)
                shift = rng.uniform(-12, 12) * np.log(10) / 10
                x[k, :n] = (feats[s:s + n].astype(np.float32) + shift - piece_ref[i]) / 5.0
                x[k, n:] = (np.log(1e-10) - piece_ref[i]) / 5.0
                y[k, :n] = targets[s:s + n]
                m[k, :n] = 1
            x, y, m = (torch.from_numpy(a).to(device) for a in (x, y, m))
            logit, value, level_logits = model(x)
            opened = (y > 0).float()
            loss = (F.binary_cross_entropy_with_logits(logit, opened, reduction='none') * m).sum() / m.sum()
            regress = opened
            if level_logits is not None:
                cls = level_targets(y)
                ce = F.cross_entropy(level_logits.reshape(-1, len(LEVELS) + 1), cls.reshape(-1), reduction='none')
                loss = loss + (ce * opened.reshape(-1)).sum() / opened.sum().clamp(min=1)
                regress = opened * (cls == len(LEVELS)).float()
            loss = loss + (F.l1_loss(value, y, reduction='none') * regress).sum() / regress.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()
        model.eval()
        res = score([(targets[s:s + n], predict(model, feats[s:s + n].astype(np.float32), device)) for _, _, s, n in val])
        print(f'epoch {epoch + 1:2d} loss {total / (len(pieces) // args.batch):.4f} '
              f'val MAE {res["mae"]:.4f} open {res["open"]:.4f} corr {res["corr"]:.4f} ({time.time() - t0:.0f} s)', flush=True)
        if best is None or res['mae'] < best['mae']:
            best = res
            torch.save(model.state_dict(), os.path.join(args.out, 'model.pt'))
    base = score([(targets[s:s + n], baseline(feats[s:s + n].astype(np.float32))) for _, _, s, n in val])
    print(f'baseline on val: MAE {base["mae"]:.4f} open {base["open"]:.4f} corr {base["corr"]:.4f}')
    print(f'best model on val: MAE {best["mae"]:.4f} open {best["open"]:.4f} corr {best["corr"]:.4f}')
    json.dump({'baseline': base, 'model': best}, open(os.path.join(args.out, 'scores.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
