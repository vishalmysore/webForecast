# model-config/

Browser inference artifacts for the from-scratch TimesFM 70M model.

| file | committed | notes |
|------|-----------|-------|
| `timesfm_small.int8.onnx` | ✅ (~65 MB) | int8-dynamic quantized graph, served to onnxruntime-web |
| `ts_meta.json` | ✅ | arch + patch/horizon/context + `point_channel` for the worker |
| `README.md` | ✅ | this file |

The **full-precision ONNX (259 MB)** and the **raw safetensors (271 MB)** are
`.gitignore`d — both exceed GitHub's 100 MB file limit. Regenerate them offline:

```bash
# 1. fetch the real weights (271 MB) from HuggingFace
curl -L -o ../../weights/model.safetensors \
  https://huggingface.co/FareedKhan/timesfm-from-scratch-70m/resolve/main/model.safetensors

# 2. export ONNX + int8 quantize (writes into this directory)
cd ../../model && python export_onnx.py \
  --weights ../weights/model.safetensors \
  --out ../docs/model-config/timesfm_small.onnx
python -c "from onnxruntime.quantization import quantize_dynamic, QuantType; \
  quantize_dynamic('../docs/model-config/timesfm_small.onnx', \
  '../docs/model-config/timesfm_small.int8.onnx', weight_type=QuantType.QInt8)"
```

int8 is near-lossless for this model (MAE ≈ 0.11 on a 40–77 series, ~0.18 %).

`timeSeriesWorker.js` reads `ts_meta.json`, loads the int8 ONNX via
onnxruntime-web (WebGPU, wasm fallback), and reproduces the autoregressive q50
decode in JS. If the file or runtime is unavailable it falls back to the
dependency-free local forecaster.
