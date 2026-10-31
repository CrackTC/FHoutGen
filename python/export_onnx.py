"""Exports a run's model to ONNX for the .NET generator: export_onnx.py <run dir> <out.onnx>.

Input "features" (1, T, N_MELS + 1): the per-cue normalised frames (train.normalise). Outputs "open_logit" (1, T) and
"value" (1, T); the generator takes the value where open_logit > 0, else 0 (train.predict without levels).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as fx  # noqa: E402
from train import load_model  # noqa: E402


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logit, value, _ = self.model(x)
        return logit, value


def main():
    run, out = sys.argv[1], sys.argv[2]
    model = load_model(run, 'cpu')
    if model.levels:
        raise SystemExit('models with the level head are not exported')
    x = torch.zeros(1, 120, fx.N_MELS + 1)
    torch.onnx.export(Wrapper(model).eval(), (x,), out, input_names=['features'], output_names=['open_logit', 'value'],
                      dynamic_axes={'features': {1: 'frames'}, 'open_logit': {1: 'frames'}, 'value': {1: 'frames'}},
                      opset_version=17, dynamo=False)
    print(f'{out}: {os.path.getsize(out) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
