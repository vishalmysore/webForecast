# Running a 70M-Parameter Time-Series Foundation Model Entirely in the Browser

*How I took a from-scratch TimesFM model, compiled it for WebGPU, and shipped a
zero-server forecasting app on GitHub Pages — plus the practical things you can
build on top of it.*

**Live demo:** https://vishalmysore.github.io/webForecast/
**Code:** https://github.com/vishalmysore/webForecast

---

## TL;DR

A real 67.8M-parameter **TimesFM-style time-series foundation model** now runs
*entirely inside a web browser* — no backend, no API keys, no per-request cost.
You load a CSV or JSON series, and the model produces a probabilistic forecast
(median plus a q10–q90 uncertainty band) on your own GPU via **WebGPU**, in about
a second. The whole thing is static files on GitHub Pages.

This post covers what it is, how the model gets from PyTorch into a browser, the
one big engineering decision that made it practical, and a set of real
applications you can build with it.

> **Credit where it's due.** The model — architecture, training, and the 70M
> checkpoint — is the work of **[Fareed Khan](https://github.com/FareedKhan-dev)**,
> from his excellent *[timesfm-from-scratch](https://github.com/FareedKhan-dev/timesfm-from-scratch)*
> project and the article
> *[Building a 200M-parameter time-series LLM from scratch](https://levelup.gitconnected.com/building-a-200m-parameter-time-series-llm-from-scratch-a99ec624ba15)*.
> This project packages that model for browser inference. It's MIT-licensed, and
> so is this.

---

## Why run a forecasting model in the browser at all?

We're used to models living behind an API. For time-series forecasting
specifically, client-side execution is unusually compelling:

- **Privacy.** Sensor data, health metrics, revenue numbers, and usage logs are
  sensitive. If the model runs on the device, *the data never leaves it*. No
  upload, no third-party processor, no data-retention policy to write.
- **Zero marginal cost.** There's no inference server to pay for. A forecast is
  free whether you have 10 users or 10 million — you're spending *their* GPU, not
  yours.
- **Offline and edge.** A Progressive Web App can forecast on a factory floor, a
  research vessel, or a plane with no connectivity.
- **Latency.** No network round-trip. The model is already in the page.

The catch has always been getting a non-trivial model to run in a browser. That's
the interesting part.

---

## The model: TimesFM, from scratch

This isn't a text LLM with numbers bolted on. It's a **decoder-only patched
transformer** in the TimesFM-2.5 lineage (Google's time-series foundation model),
reimplemented from scratch. The "small" tier — the one published on Hugging Face
and used here — is:

| property | value |
|---|---|
| parameters | **67,810,176** (67.8M) |
| model dim | 1024 |
| layers | 10 |
| heads | 16 |
| input patch | 32 time-steps |
| output horizon | 128 time-steps per position |
| context | 512 time-steps (16 patches) |
| outputs | point forecast + **9 quantiles** |

A few architectural details matter for what follows:

- **Patching instead of tokenizing.** Continuous values are grouped into patches
  of 32 points. Each patch (plus a padding mask) is projected to the model
  dimension by a small residual block. This is the "token" — no vocabulary, no
  embedding table.
- **First-patch RevIN.** The series is reversibly normalized using the statistics
  of its first valid patch, which is what lets one model generalize across
  wildly different scales (electricity in MW, temperature in °C, stock prices in
  dollars).
- **Quantile heads.** The model emits 9 quantiles, so you get calibrated
  uncertainty — a *band*, not just a line. It forecasts autoregressively by
  feeding the **median (q50)** back into the context.
- **Modern internals:** sandwich RMSNorm, fused QKV with qk-norm, RoPE, a
  per-dimension learned attention scale, and a two-matrix SiLU feed-forward.

That last point is exactly why the "obvious" browser path doesn't work.

---

## The decision that made it practical: ONNX, not MLC

The best-known way to run models on WebGPU in a browser is **MLC LLM / WebLLM**
(Apache TVM). It's superb — for the architectures it ships definitions for:
LLaMA, Qwen, Mistral, Phi. Those are all *text* models.

MLC's compiler doesn't build a model from a JSON description of "layers and
heads." It needs the architecture implemented in its **TVM-Relax model DSL** and
registered in the toolchain. TimesFM's patched decoder — with fused QKV, qk-norm,
per-dim scale, quantile heads, and a patch tokenizer over values+mask — is none
of the built-ins. Going the MLC route would mean *authoring a new TVM model
definition*, making the weight converter line up, **and** bypassing WebLLM's
token-oriented, KV-cache chat runtime to inject continuous patch tensors by hand.
That's compiler engineering, not a config file.

The pragmatic alternative is **ONNX Runtime Web**, which has a first-class WebGPU
execution provider and doesn't care whether your model is "text-shaped":

```
PyTorch model  ──torch.onnx.export──▶  ONNX graph  ──int8 quantize──▶  65 MB .onnx
                                                                            │
                                              onnxruntime-web (WebGPU EP) ◀──┘
```

Export the model *once*, quantize, and run it on WebGPU with a few lines of
JavaScript. No per-architecture compiler work, no tokenizer assumptions. For a
custom architecture, this is the shorter and more robust path.

### The pipeline in detail

1. **Load the real weights.** The published 70M `safetensors` loads into the
   faithful PyTorch reimplementation with an *exact* state-dict match — same
   layer names, same shapes, zero surprises.
2. **Export `encode` to ONNX.** Only the single forward pass is exported (fixed
   512-context → per-patch `[16, 128, 10]` predictions plus the RevIN
   `mu`/`sigma`). The autoregressive loop stays in JavaScript, which keeps the
   graph fixed-shape and WebGPU-friendly.
   - *Gotcha worth noting:* `torch.eye`/`torch.tril` in the attention mask export
     as `EyeLike`/`Trilu`, which onnxruntime-web doesn't implement. Rebuilding the
     causal mask with `arange` comparisons produces an identical result and a
     fully compatible graph.
3. **int8 dynamic quantization.** The fp32 ONNX is 259 MB — over GitHub's 100 MB
   file limit. int8 brings it to **65 MB** with a forecast MAE of ~0.11 on a
   40–77 series (≈0.18%) — invisible on a chart, and it fits in the repo.
4. **Run in a Web Worker.** onnxruntime-web loads the model on the WebGPU backend
   (falling back to a WASM CPU backend, then to a dependency-free JS forecaster
   if the model can't load). The worker reproduces the rest of the forecasting
   loop in JS: build the padding mask, denormalize with `mu`/`sigma`, take the
   last patch's 128-step horizon, feed q50 back for longer horizons, and surface
   q10/q90 as the uncertainty band.

### Does it actually match the original?

| check | result |
|---|---|
| Parameter count | 67,810,176 — matches the model card exactly |
| ONNX vs. PyTorch | max difference **1.4e-6** |
| int8 vs. full precision | forecast MAE **0.11** (~0.18%) |
| In-browser run | **WebGPU**, ~1.3 s per 128-step forecast |

The model that runs in your tab is, to within float precision, the same model
Fareed trained.

---

## What you can build with this

The interesting question isn't the demo — it's what a free, private, offline,
GPU-accelerated forecaster unlocks as a *building block*. A few concrete
directions:

### 1. Privacy-first personal analytics
Wearables and health apps generate deeply personal series: heart rate, sleep,
glucose, weight, steps. Forecasting them normally means shipping that data to a
server. Here, **the data never leaves the phone**. Build a "your next 7 days"
projection for a fitness or health PWA with no backend and nothing to breach.

### 2. Backendless dashboards and BI widgets
Drop a forecast overlay onto any chart in an internal dashboard — traffic,
signups, revenue, inventory — with *no forecasting service to stand up*. The
CSV/JSON the dashboard already has becomes a forecast, client-side. Great for
embedded analytics in SaaS products where you don't want per-customer inference
bills.

### 3. Anomaly detection via the uncertainty band
Because the model emits quantiles, you get a free anomaly detector: when the
*actual* next value falls outside the predicted **q10–q90** band, flag it. Run it
in the browser over streaming metrics (latency, error rates, temperatures) and
alert without sending telemetry anywhere.

### 4. Edge and offline forecasting
Field equipment, IoT gateways, research instruments, agricultural sensors — a PWA
that caches the model can forecast **completely offline**. Solar generation, tank
levels, soil moisture, energy load: all forecastable on-site with no cloud
dependency.

### 5. DevOps / capacity planning
Point it at CPU, memory, request-rate, or queue-depth history to project the next
few hours and pre-scale — a lightweight, client-side companion to your monitoring
UI. The included `network_traffic_5min.json` and `electricity_hourly.csv` samples
mimic exactly this shape.

### 6. An embeddable "forecast this column" component
Wrap the worker as a reusable widget: hand it a numeric array, get back
`{point, lo, hi}`. Now any spreadsheet tool, notebook, or CMS can offer
"forecast this" with a one-line integration and zero server cost. (The app's
`timeSeriesWorker.js` is already structured as exactly this interface.)

### 7. Teaching and interactive exploration
A foundation model you can poke at live — change the horizon, swap datasets, watch
the uncertainty band widen on a random walk versus tighten on a clean seasonal
signal — is a genuinely good way to build intuition about probabilistic
forecasting. The bundled samples span trend-only, multiplicative seasonality,
daily on/off (solar), long cycles (sunspots), and a deliberately hard
non-seasonal random walk (`stock_price_daily.json`).

> A note on the financial example: a random walk is *supposed* to be hard, and
> the model correctly responds with a flat, wide-banded forecast. That's a
> feature — it's honestly expressing "I don't know." This is a demo of model
> behavior, **not** investment advice.

---

## Try it

1. Open **https://vishalmysore.github.io/webForecast/**
2. Click a **sample dataset** (or upload your own CSV/JSON), set a horizon, and
   hit **Forecast**.
3. Check the status bar — `timesfm-70m-onnx-webgpu` means it ran on your GPU.

### The sample datasets

The app ships with **ten ready-to-use series**, deliberately chosen to span the
patterns a forecaster meets in the wild — and to show where the model shines and
where it (honestly) struggles. Click any of them on the page to load and forecast
instantly, hit the **⭳** to download the raw file, or grab and edit them from
[`docs/samples/`](https://github.com/vishalmysore/webForecast/tree/main/docs/samples).

| dataset | format | pattern it demonstrates |
|---|---|---|
| `electricity_hourly.csv` | CSV | daily **and** weekly seasonality + slow trend |
| `solar_generation_hourly.csv` | CSV | daytime on/off "bell" (zeros at night) + weather noise |
| `network_traffic_5min.json` | JSON | intraday cycle with occasional bursts |
| `web_visits.json` | JSON | weekday/weekend traffic rhythm |
| `temperature_daily.csv` | CSV | annual (365-day) seasonality |
| `stock_price_daily.json` | JSON | **non-seasonal random walk** — the hard case |
| `retail_sales_weekly.csv` | CSV | yearly seasonality + trend + a December spike |
| `co2_monthly.csv` | CSV | strong upward trend + annual cycle (a Keeling curve) |
| `airline_passengers_monthly.csv` | CSV | *multiplicative* growth × seasonality |
| `sunspots_monthly.csv` | CSV | a long (~11-year) cycle |

Two are worth loading back-to-back to *see* what a probabilistic model does:

- **`co2_monthly.csv`** — a clean trend + seasonal signal. The forecast confidently
  continues both the rise and the annual wiggle, with a **tight** q10–q90 band.
- **`stock_price_daily.json`** — a random walk with no real structure. The model
  responds with a nearly flat projection and a **wide, fanning** band. That's the
  correct behavior: it's saying "I don't know," out loud, through its uncertainty.

That contrast is the whole point of quantile forecasting — the *band* tells you
how much to trust the *line*.

**Input formats.** CSV uses the last numeric column of each row (so a
`timestamp,value` file just works); JSON is a plain array of numbers. Bring your
own by clicking **Load CSV/JSON** — the series is normalized per-instance, so any
scale or unit is fine.

---

## Honest limitations

- **It's a demo of a small model.** 70M parameters is the "small" tier; the base
  TimesFM is larger and stronger. Zero-shot forecasts are good on clean seasonal
  and trending data and weak on genuinely unpredictable series (see the FX/random
  walk note in the original repo).
- **First load downloads 65 MB.** It's cached afterward, but the initial fetch
  isn't free on a phone data plan.
- **WebGPU availability.** Modern Chrome/Edge are solid; Firefox and Safari are
  catching up. Without WebGPU it falls back to a WASM CPU path (slower) or a
  simple built-in forecaster.
- **Not financial or medical advice.** It's a general-purpose forecaster; treat
  its outputs accordingly.

---

## Credits & license

- **Model, training, and weights:** [Fareed Khan](https://github.com/FareedKhan-dev)
  — [timesfm-from-scratch](https://github.com/FareedKhan-dev/timesfm-from-scratch)
  · [70M checkpoint](https://huggingface.co/FareedKhan/timesfm-from-scratch-70m)
- **Runtime:** [onnxruntime-web](https://onnxruntime.ai/) (WebGPU)
- **This deployment:** [webForecast](https://github.com/vishalmysore/webForecast)

Everything here is **MIT-licensed**. The vendored model code retains Fareed
Khan's copyright notice; the browser packaging, export tooling, and samples are
under this project's MIT license. You're free to use, modify, and publish it —
just keep the attribution.
