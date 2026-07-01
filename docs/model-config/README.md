# model-config/

Compiled WebGPU artifacts land here. This directory is intentionally empty in
git — the ~100MB of weights are produced by the offline build, not committed by
hand.

Expected contents after `scripts/compile.sh` (spec Section 3 & 4):

```
model-config/
├── ts_model_webgpu.wasm      # TVM/WebGPU orchestration binary
├── mlc-chat-config.json      # runtime config (context, sampling)
├── ts_io.safetensors         # patch-embed + forecast head (runtime-side)
├── ts_meta.json              # patch_size / ctx / horizon / norm scheme
└── params/
    ├── params_shard_0.bin    # chunked weights, each < 50MB (GitHub 100MB limit)
    ├── params_shard_1.bin
    └── ...
```

Until these exist, `timeSeriesWorker.js` detects the missing `ts_meta.json` and
transparently runs the on-device JavaScript fallback engine, so the page still
forecasts. Once staged, the worker auto-selects the WebGPU engine.

If the repo approaches GitHub's 1GB quota, push `params/` to a public Hugging
Face model repo and point `CFG.modelDir` at its CDN URL instead (spec 5).
```
