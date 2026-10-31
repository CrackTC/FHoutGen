#!/usr/bin/env bash
# Features of every local fhout sheet, a sixth at a time (the wavs of a part are deleted once its features are cached):
# data/features_part<k>/ for k = 1..5 (part 0 is data/features, from data/pairs), then one merged cache data/features_full.
set -euo pipefail
cd "$(dirname "$0")"
STREAMING=../demo/Assets/StreamingAssets
EXTRACT=Extract/bin/Release/net10.0/fhout-extract.exe
PY=.venv/Scripts/python.exe
for k in 1 2 3 4 5; do
    if [ -f "data/features_part$k/index.json" ]; then echo "part $k cached"; continue; fi
    rm -rf data/batch
    "$EXTRACT" export --streaming "$STREAMING" --out data/batch --every 6 --offset "$k" | tail -1
    "$PY" -I python/features.py data/batch "data/features_part$k" | tail -1
    rm -rf data/batch
done
"$PY" -I python/merge_features.py data/features_full data/features data/features_part1 data/features_part2 \
    data/features_part3 data/features_part4 data/features_part5
