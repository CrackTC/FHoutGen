"""Merges feature caches (features.py output) into one: merge_features.py <out> <cache> [<cache> ...]."""
import json
import os
import sys

import numpy as np


def main():
    out, parts = sys.argv[1], sys.argv[2:]
    os.makedirs(out, exist_ok=True)
    feats = [np.load(os.path.join(p, 'feats.npy'), mmap_mode='r') for p in parts]
    total = sum(len(f) for f in feats)
    merged = np.lib.format.open_memmap(os.path.join(out, 'feats.npy'), mode='w+', dtype=np.float16,
                                       shape=(total, feats[0].shape[1]))
    targets, index = [], []
    start = 0
    for p, f in zip(parts, feats):
        merged[start:start + len(f)] = f
        targets.append(np.load(os.path.join(p, 'targets.npy')))
        for sheet, cue, s, n in json.load(open(os.path.join(p, 'index.json'))):
            index.append([sheet, cue, s + start, n])
        start += len(f)
    merged.flush()
    np.save(os.path.join(out, 'targets.npy'), np.concatenate(targets))
    json.dump(index, open(os.path.join(out, 'index.json'), 'w'))
    print(f'{len(index)} cues, {total} frames ({total / 60 / 3600:.1f} h)')


if __name__ == '__main__':
    main()
