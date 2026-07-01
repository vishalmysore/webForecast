"""Load the published 70M TimesFM weights and export a static ONNX graph for
onnxruntime-web (WebGPU). Spec goal: run the *real* model in the browser.

We export only `encode` (one forward: context+padding -> per-position horizon
predictions + first-patch mean/sigma). The autoregressive loop, RevIN denorm,
and q50 feedback are reproduced in JS (docs/timeSeriesWorker.js), so the graph
stays fixed-shape and WebGPU-friendly.

    python export_onnx.py \
        --weights weights/model.safetensors \
        --out docs/model-config/timesfm_small.onnx \
        [--fp16]

Fixed I/O (batch 1, context 512 = 16 patches of 32):
    inputs : x [1, 512] float32, padding [1, 512] float32 (1 = pad/missing)
    outputs: out [1, 16, 128, 10] float32, mu [1], sigma [1]
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from safetensors.torch import load_file

from config import small
from timesfm import build_model, count_params


class EncodeForExport(nn.Module):
    """Flattens encode's nested tuple so ONNX emits three named outputs."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, padding):
        out, (mu, sigma) = self.model.encode(x, padding)
        return out, mu, sigma


def load_weights(model, path):
    sd = load_file(path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")
    if not missing and not unexpected:
        print("  state_dict matched exactly")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/model.safetensors")
    ap.add_argument("--out", default="docs/model-config/timesfm_small.onnx")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    cfg = small()
    model = build_model(cfg).eval()
    print(f"built small model: {count_params(model):,} params")

    if os.path.exists(args.weights):
        load_weights(model, args.weights)
    else:
        print(f"  [warn] weights not found at {args.weights}; exporting random init")

    wrapper = EncodeForExport(model).eval()
    L = cfg.context_len
    x = torch.randn(1, L)
    pad = torch.zeros(1, L)

    # Sanity forward before export.
    with torch.no_grad():
        out, mu, sigma = wrapper(x, pad)
    print(f"forward ok: out={tuple(out.shape)} mu={tuple(mu.shape)} sigma={tuple(sigma.shape)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.onnx.export(
        wrapper, (x, pad), args.out,
        input_names=["x", "padding"],
        output_names=["out", "mu", "sigma"],
        opset_version=args.opset,
        dynamo=False,
        do_constant_folding=True,
    )
    print(f"exported ONNX -> {args.out}")

    if args.fp16:
        from onnxconverter_common import float16
        import onnx
        m = onnx.load(args.out)
        m16 = float16.convert_float_to_float16(m, keep_io_types=True)
        onnx.save(m16, args.out)
        print("converted graph weights to fp16")

    meta = {
        "arch": "timesfm-patched-decoder-small",
        "params": count_params(model),
        "model_dim": cfg.model_dim,
        "num_layers": cfg.num_layers,
        "num_heads": cfg.num_heads,
        "patch_len": cfg.patch_len,
        "horizon_len": cfg.horizon_len,
        "context_len": cfg.context_len,
        "num_outputs": cfg.num_outputs,
        "quantiles": list(cfg.quantiles),
        "point_channel": 5,  # q50 median feedback (TimesFM-2.5 default)
        "onnx": os.path.basename(args.out),
    }
    meta_path = os.path.join(os.path.dirname(args.out), "ts_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
