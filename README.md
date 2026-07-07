# webForecast

The **from-scratch TimesFM 70M** time-series foundation model, running **entirely
in the browser** on WebGPU — no server, no API. Load a series, get an
autoregressive probabilistic forecast (q50 point + q10–q90 band) rendered on a
canvas.

**Live:** https://vishalmysore.github.io/webForecast/ · **Write-up:** [article.md](article.md) · **License:** [MIT](LICENSE)

Built on and faithful to **[FareedKhan-dev/timesfm-from-scratch](https://github.com/FareedKhan-dev/timesfm-from-scratch)**
(model + article by Fareed Khan). The published checkpoint
[FareedKhan/timesfm-from-scratch-70m](https://huggingface.co/FareedKhan/timesfm-from-scratch-70m)
is loaded, exported to ONNX, int8-quantized, and served to onnxruntime-web.

## Architecture (the "small" / 70M tier)

TimesFM-2.5-style **patched decoder** — not a text LLM. Verified param count:
**67,810,176 (67.8M)**.

| field | value |
|-------|-------|
| model_dim | 1024 |
| layers | 10 |
| heads | 16 |
| patch_len (in) | 32 |
| horizon_len (out) | 128 |
| context | 512 (16 patches) |
| tokenizer | `ResidualBlock(2·patch → dim)` — values **+ padding mask** |
| norm | **sandwich RMSNorm**, qk-norm, per-dim learned scale |
| attention | fused QKV, RoPE, SDPA (scale=1.0) |
| FFN | two-matrix SiLU (not SwiGLU) |
| output | point + **9 quantile** heads (10 channels); AR feedback on q50 |
| RevIN | **first-patch** reversible instance norm |

## Layout

```
webForecast/
├── model/                    # faithful PyTorch (vendored from the reference)
│   ├── config.py             # tiny / small / base configs
│   ├── layers.py             # ResidualBlock, RMSNorm, RoPE, attention, block
│   ├── revin.py              # first-patch RevIN
│   ├── timesfm.py            # PatchedDecoder (encode + AR forecast)
│   └── export_onnx.py        # load HF weights -> static ONNX for the browser
├── docs/                     # GitHub Pages site
│   ├── index.html, style.css, main.js
│   ├── timeSeriesWorker.js   # onnxruntime-web (WebGPU) + JS AR decode + fallback
│   ├── model-config/
│   │   ├── timesfm_small.int8.onnx   # 65MB int8 graph (committed)
│   │   └── ts_meta.json
│   └── wc/                   # World Cup 2026 champion-odds demo (see below)
│       ├── index.html, wc.css, wc.js
│       ├── poisson.js        # Dixon-Coles match model
│       ├── tournament.js     # 48-team Monte-Carlo (groups → knockout)
│       └── elo.json          # real Elo + the 12 groups (precomputed)
├── scripts/
│   └── build_elo.py          # offline Elo + group precompute -> docs/wc/elo.json
├── requirements.txt
└── spec.md                   # original system spec (kept for provenance)
```

## Reproduce the model artifacts (offline)

```bash
pip install -r requirements.txt

# sanity: build the 70M model, print param count + a forecast
python model/timesfm.py

# fetch real weights (271MB) and export the browser ONNX (see docs/model-config/README.md)
python model/export_onnx.py --weights weights/model.safetensors \
  --out docs/model-config/timesfm_small.onnx
```

## Run locally

```bash
npx serve docs -p 5187      # http://localhost:5187
```

The worker loads the int8 ONNX and runs the real model on WebGPU (falling back to
the wasm CPU backend, then to a dependency-free local forecaster if the model or
runtime can't load). The status bar reports which engine ran, e.g.
`timesfm-70m-onnx-webgpu`.

## How the browser inference works

The ONNX graph is the model's `encode` (fixed 512-context forward → per-patch
`[16, 128, 10]` predictions in normalized space + first-patch `mu, sigma`).
`timeSeriesWorker.js` reproduces the rest of `PatchedDecoder.forecast()` in JS:
build the padding mask, denormalize with `mu, sigma`, take the last patch's
128-step horizon, feed the **q50 median** (channel 5) back autoregressively for
horizons > 128, and surface **q10/q90** (channels 1/9) as the uncertainty band.

## Verified

- ✅ Model builds to **67.8M** params; `strict=True` load of the real weights.
- ✅ ONNX matches PyTorch to **1.4e-6**; int8 forecast MAE **0.11** (~0.18 %).
- ✅ Real model runs in-browser on **WebGPU** (~1.3 s / 128-step forecast in
  local preview), forecast + band render, zero console errors.

## Bonus demo — World Cup 2026 champion odds (`docs/wc/`)

A worked example of using the forecaster as one stage of a larger pipeline, all
client-side. **[Live](https://vishalmysore.github.io/webForecast/wc/)** ·
**[Write-up: article_soccer.md](article_soccer.md).**

1. **Data → Elo.** `scripts/build_elo.py` reads ~49k international results
   (1872–2026, [martj42/international_results](https://github.com/martj42/international_results))
   and computes a World Football Elo rating per team, sampled monthly, plus the
   real 12-group 2026 draw (reconstructed from the scheduled fixtures) →
   `docs/wc/elo.json`.
2. **Forecast.** The same TimesFM worker projects each of the 48 teams' Elo
   trajectory forward; the 1-step q50 is its tournament-time strength and the
   q10/q90 band is its uncertainty.
3. **Match model.** `poisson.js` maps two Elos to a Dixon-Coles bivariate-Poisson
   scoreline distribution (even match ≈ 1.3 goals/side; a 200-Elo edge ≈ +1.4
   goal supremacy).
4. **Tournament.** `tournament.js` Monte-Carlos the real format — 12 groups of 4
   (round-robin) → top two + eight best thirds → 32-team knockout — sampling
   scorelines per match. Optionally each team's strength is drawn from its
   forecast band per trial, so forecast uncertainty propagates into the odds.

The result is a per-team championship probability (favourites land in the
realistic ~15–30 % range, and everything sums to 100 %). It's a demo, not a
betting model: the knockout seeding is a merit-based simplification of FIFA's
positional bracket, and the odds carry wide, honest uncertainty.

To regenerate the data: `curl` the results CSV (command in `.gitignore`) into
`scripts/data/`, then `python scripts/build_elo.py`.

## Deployment note

If loading onnxruntime-web from the CDN is blocked on the live site (a COEP /
service-worker interaction can do this on `*.github.io`), the page transparently
falls back to the local forecaster. Self-hosting the onnxruntime-web dist under
`docs/vendor/` and pointing `CFG.ortVersion`/`wasmPaths` at it removes that
dependency.

## Credit & license

Model architecture, training, and the 70M checkpoint are the work of
**[Fareed Khan](https://github.com/FareedKhan-dev)**
([timesfm-from-scratch](https://github.com/FareedKhan-dev/timesfm-from-scratch),
MIT). This repo packages that model for zero-server WebGPU inference.

Everything here is **MIT-licensed** ([LICENSE](LICENSE)) — the vendored model code
(`model/layers.py`, `model/revin.py`, `model/timesfm.py`) retains Fareed Khan's
copyright notice; the browser app, export tooling, and samples are under this
project's MIT license. Free to use, modify, and publish with attribution.
