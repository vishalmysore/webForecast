#!/usr/bin/env bash
# Spec Section 3, Step 1 -- quantize the exported LLaMA-format checkpoint to
# q4f16_1 (4-bit weights, FP16 scales). Brings ~400MB FP16 -> ~100-120MB.
set -euo pipefail

EXPORT_DIR="${1:-export}"          # output of model/export_llama.py
QUANT_DIR="${2:-quantized-model}"
QUANT="q4f16_1"

echo ">> converting + quantizing $EXPORT_DIR ($QUANT)"

# mlc_llm needs the HF-style config.json + model.safetensors we emitted.
mlc_llm convert_weight "$EXPORT_DIR" \
  --quantization "$QUANT" \
  --output "$QUANT_DIR"

# Generate the runtime chat config (context/sampling defaults live here).
mlc_llm gen_config "$EXPORT_DIR" \
  --quantization "$QUANT" \
  --conv-template LM \
  --context-window-size 512 \
  --output "$QUANT_DIR"

echo ">> quantized weights + mlc-chat-config.json in $QUANT_DIR/"
