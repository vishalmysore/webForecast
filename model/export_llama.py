"""Adapter that formats the 200M patched TS transformer as a LLaMA checkpoint
so that `mlc_llm` maps its components onto LLaMA structural blocks.

Spec Section 6, directive #2. Two things happen here:

1. The 12 decoder blocks are already LLaMA-shaped, so their weights are copied
   under HuggingFace LLaMA names (model.layers.N.self_attn.q_proj.weight, ...).

2. The embedding table / lm_head that a LLaMA graph *requires* are emitted as
   structural placeholders sized to the tiny `vocab_size`. The genuine input
   projection (patch_embed) and output projection (forecast head) are exported
   SEPARATELY into `ts_io.safetensors`; the frontend worker applies them around
   the compiled backbone. This is the "swap / bypass during compilation" step
   the spec calls for -- the continuous patch payload is injected in place of
   the token embedding lookup at runtime.

Usage:
    python export_llama.py --ckpt runs/ts200m.pt --out export/

Produces:
    export/
      config.json                 # HF LLaMA config for `mlc_llm convert_weight`
      model.safetensors           # backbone + placeholder embed/lm_head
      ts_io.safetensors           # patch_embed + forecast head (runtime-side)
      ts_meta.json                # patch_size, ctx, horizon, norm scheme
"""

import argparse
import json
import os

import torch
from safetensors.torch import save_file

from config import TSConfig
from ts_transformer import TimeSeriesTransformer


def to_llama_state_dict(model: TimeSeriesTransformer):
    cfg = model.cfg
    sd = model.state_dict()
    out = {}

    # Placeholder embedding + lm_head (structural, tiny vocab).
    out["model.embed_tokens.weight"] = torch.zeros(cfg.vocab_size, cfg.d_model)
    out["lm_head.weight"] = torch.zeros(cfg.vocab_size, cfg.d_model)

    for i in range(cfg.n_layers):
        p = f"layers.{i}."
        q = f"model.layers.{i}."
        for a, b in [
            ("self_attn.q_proj.weight", "self_attn.q_proj.weight"),
            ("self_attn.k_proj.weight", "self_attn.k_proj.weight"),
            ("self_attn.v_proj.weight", "self_attn.v_proj.weight"),
            ("self_attn.o_proj.weight", "self_attn.o_proj.weight"),
            ("mlp.gate_proj.weight", "mlp.gate_proj.weight"),
            ("mlp.up_proj.weight", "mlp.up_proj.weight"),
            ("mlp.down_proj.weight", "mlp.down_proj.weight"),
            ("input_layernorm.weight", "input_layernorm.weight"),
            ("post_attention_layernorm.weight", "post_attention_layernorm.weight"),
        ]:
            out[q + b] = sd[p + a].contiguous()

    out["model.norm.weight"] = sd["norm.weight"].contiguous()
    return out


def io_state_dict(model: TimeSeriesTransformer):
    sd = model.state_dict()
    return {
        "patch_embed.proj.weight": sd["patch_embed.proj.weight"].contiguous(),
        "patch_embed.proj.bias": sd["patch_embed.proj.bias"].contiguous(),
        "head.proj.weight": sd["head.proj.weight"].contiguous(),
        "head.proj.bias": sd["head.proj.bias"].contiguous(),
    }


def hf_llama_config(cfg: TSConfig) -> dict:
    """Minimal HF LLaMA config understood by mlc_llm's llama model definition."""
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": cfg.d_model,
        "intermediate_size": cfg.intermediate_size,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_kv_heads,
        "max_position_embeddings": cfg.n_ctx_patches,
        "rms_norm_eps": cfg.rms_norm_eps,
        "rope_theta": cfg.rope_theta,
        "vocab_size": cfg.vocab_size,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="trained .pt state_dict (random init if omitted)")
    ap.add_argument("--out", default="export")
    ap.add_argument("--fp16", action="store_true", help="cast tensors to float16 before saving")
    args = ap.parse_args()

    cfg = TSConfig()
    model = TimeSeriesTransformer(cfg)
    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
        print(f"loaded checkpoint: {args.ckpt}")
    else:
        print("WARNING: no --ckpt given, exporting randomly initialized weights")

    os.makedirs(args.out, exist_ok=True)

    backbone = to_llama_state_dict(model)
    io = io_state_dict(model)
    if args.fp16:
        backbone = {k: v.half() for k, v in backbone.items()}
        io = {k: v.half() for k, v in io.items()}

    save_file(backbone, os.path.join(args.out, "model.safetensors"))
    save_file(io, os.path.join(args.out, "ts_io.safetensors"))

    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(hf_llama_config(cfg), f, indent=2)

    meta = {
        "patch_size": cfg.patch_size,
        "n_ctx_patches": cfg.n_ctx_patches,
        "n_pred_patches": cfg.n_pred_patches,
        "horizon": cfg.horizon,
        "d_model": cfg.d_model,
        "normalization": "instance_revin",
        "io_weights": "ts_io.safetensors",
    }
    with open(os.path.join(args.out, "ts_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    n = sum(p.numel() for p in model.parameters())
    print(f"exported ~{n/1e6:.1f}M-param model to {args.out}/")
    print("next: run scripts/quantize.sh then scripts/compile.sh")


if __name__ == "__main__":
    main()
