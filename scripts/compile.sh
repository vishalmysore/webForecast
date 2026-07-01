#!/usr/bin/env bash
# Spec Section 3, Step 2 -- compile the quantized graph to a WebGPU WASM module
# and stage the chunked weight shards under docs/model-config/ for GitHub Pages.
set -euo pipefail

QUANT_DIR="${1:-quantized-model}"
DOCS_MODEL="${2:-docs/model-config}"
WASM_OUT="$DOCS_MODEL/ts_model_webgpu.wasm"

mkdir -p "$DOCS_MODEL"

echo ">> compiling $QUANT_DIR -> $WASM_OUT (target: webgpu)"
mlc_llm compile "$QUANT_DIR/mlc-chat-config.json" \
  --device webgpu \
  --output "$WASM_OUT"

# Stage weights + config next to the wasm. MLC already shards params under
# GitHub's 100MB/file limit (spec 1: shards capped well under 50MB).
echo ">> staging weights + config into $DOCS_MODEL/"
cp -r "$QUANT_DIR/params" "$DOCS_MODEL/params"
cp "$QUANT_DIR/mlc-chat-config.json" "$DOCS_MODEL/mlc-chat-config.json"

# The continuous patch-embed + forecast head run runtime-side (outside the
# compiled token graph); ship them alongside if present.
if [ -f export/ts_io.safetensors ]; then cp export/ts_io.safetensors "$DOCS_MODEL/"; fi
if [ -f export/ts_meta.json ]; then cp export/ts_meta.json "$DOCS_MODEL/"; fi

echo ">> done. Largest shard:"
du -h "$DOCS_MODEL/params"/*.bin 2>/dev/null | sort -h | tail -1 || true
echo ">> commit docs/ and enable GitHub Pages from /docs (spec Section 5)."
