# webForecast

Client-side, zero-server **200M-parameter patched time-series transformer** that
runs forecasting entirely in the browser via WebGPU. Implements
[`spec.md`](spec.md) — a decoder-only, LLaMA-structured model compiled with the
MLC LLM (TVM Unity) toolchain and hosted statically on GitHub Pages.

## Layout

```
webForecast/
├── spec.md                     # the authoritative system spec
├── requirements.txt            # torch / safetensors / mlc-llm
├── model/
│   ├── config.py               # TSConfig — all architecture numbers (~201M)
│   ├── ts_transformer.py       # the model: patch-embed + LLaMA blocks + head
│   └── export_llama.py         # adapter -> HF-LLaMA checkpoint for mlc_llm
├── scripts/
│   ├── quantize.sh             # spec 3.1  q4f16_1 quantization
│   └── compile.sh              # spec 3.2  WebGPU compile + stage into docs/
└── docs/                       # the GitHub Pages site (spec 4)
    ├── index.html
    ├── style.css
    ├── main.js                 # UI: ingest, worker handshake, canvas chart
    ├── timeSeriesWorker.js     # inference pipeline (WebGPU + local fallback)
    └── model-config/           # compiled weights land here (git-empty)
```

## Architecture (spec §2)

| field | value |
|-------|-------|
| type | decoder-only, LLaMA-structured (RMSNorm, RoPE, SwiGLU) |
| d_model | 1024 |
| layers | 12 |
| heads | 16 |
| context | 512 patches |
| patch size P | 16 time-steps |
| params | ~201.4M |
| quantization | q4f16_1 (~100–120 MB) |

The single adaptation vs. a text LLM: the `nn.Embedding` token table is replaced
by a **linear patch projection** (`P → d_model`) and the `lm_head` by a **forecast
head** (`d_model → P`). Everything between is a stock LLaMA graph, so MLC compiles
it to WebGPU without custom operators. Instance normalization (RevIN-style) wraps
the whole forward pass.

## Build the model (offline, one-time, needs a GPU)

```bash
pip install -r requirements.txt

# 1. sanity-check the architecture / param count
python model/ts_transformer.py

# 2. (train your checkpoint here -> runs/ts200m.pt)

# 3. export to a LLaMA-format checkpoint for MLC
python model/export_llama.py --ckpt runs/ts200m.pt --out export --fp16

# 4. quantize + compile to WebGPU, staged into docs/model-config/
bash scripts/quantize.sh export quantized-model
bash scripts/compile.sh  quantized-model docs/model-config
```

## Run the site

Serve `docs/` over HTTP (WebGPU + workers need a real origin):

```bash
python -m http.server -d docs 8000   # http://localhost:8000
```

- **With compiled weights staged** → the worker loads the MLC WebGPU model.
- **Without them** → the worker falls back to a dependency-free on-device
  forecaster running the *same* normalize→patch→autoregress→denormalize
  pipeline, so the demo works immediately. The status bar always reports which
  engine ran.

## Deploy

Push to `main`, enable **GitHub Pages → deploy from `/docs`** (spec §5). If the
weights push the repo toward GitHub's 1 GB limit, host `params/` on a public
Hugging Face model repo and repoint `CFG.modelDir` in `timeSeriesWorker.js`.

## Honest status

- ✅ Model definition, param budget, and LLaMA export adapter are complete and
  verified (`python model/ts_transformer.py`).
- ✅ Frontend pipeline (ingest, instance-norm, patching, denorm, charting) and
  the on-device fallback engine are complete and runnable now.
- ⏳ The WebGPU engine loads MLC artifacts, but driving the compiled graph with
  *continuous* patch tensors needs the low-level TVM runtime bind (web-llm's
  chat API is token-oriented). That harness is stubbed with a clear error and
  is the remaining integration step once real weights exist — see the note in
  `timeSeriesWorker.js`.
