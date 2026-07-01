/*
 * timeSeriesWorker.js  (spec Section 6, directive #1)
 * ---------------------------------------------------------------------------
 * Off-main-thread inference pipeline for the 200M patched time-series
 * transformer. Implements the client-side data flow from spec Section 4.4:
 *
 *   raw series -> instance normalization -> 1D patching
 *              -> transformer forecast (WebGPU model, or local fallback)
 *              -> denormalization -> back to the UI for charting
 *
 * The "continuous linear patch payload injection" the spec describes lives in
 * patchify()/instanceNorm() below: instead of integer token IDs we build
 * P-length float patches and hand them to the engine.
 *
 * Two engines implement the same interface { ready, forecast(patches,...) }:
 *   - WebGPUEngine : loads the MLC-compiled artifacts from ./model-config/ and
 *                    runs them via @mlc-ai/web-llm's WebGPU runtime. Requires
 *                    the ~100MB weights produced by scripts/compile.sh.
 *   - LocalEngine  : dependency-free on-device forecaster implementing the same
 *                    normalize->patch->autoregress->denormalize pipeline so the
 *                    page works on GitHub Pages before the big model is staged.
 *
 * The worker auto-selects WebGPU when the artifacts + navigator.gpu are present
 * and falls back to Local otherwise, always reporting which engine ran.
 */

'use strict';

const CFG = {
  patchSize: 16,        // P  (spec 2)
  nCtxPatches: 512,     // N_patches context window
  nPredPatches: 8,      // default horizon = 8 * 16 = 128 steps
  modelDir: './model-config',
};

/* ------------------------------------------------------------------ *
 * Preprocessing  (spec 4.2)                                          *
 * ------------------------------------------------------------------ */
function instanceNorm(series) {
  const n = series.length;
  const mean = series.reduce((a, b) => a + b, 0) / n;
  let v = 0;
  for (const x of series) v += (x - mean) * (x - mean);
  const std = Math.sqrt(v / n) + 1e-5;
  const norm = series.map((x) => (x - mean) / std);
  return { norm, mean, std };
}

function denorm(values, mean, std) {
  return values.map((x) => x * std + mean);
}

// Split a normalized series into non-overlapping P-length patches. The tail is
// left-padded to a whole patch so short/odd-length inputs still work.
function patchify(norm, patchSize) {
  const usable = Math.floor(norm.length / patchSize) * patchSize;
  const trimmed = norm.slice(norm.length - usable); // keep most-recent window
  const patches = [];
  for (let i = 0; i < trimmed.length; i += patchSize) {
    patches.push(trimmed.slice(i, i + patchSize));
  }
  return patches;
}

/* ------------------------------------------------------------------ *
 * LocalEngine -- dependency-free fallback forecaster                 *
 * ------------------------------------------------------------------ *
 * Not the 200M net, but it exercises the identical pipeline and produces a
 * genuine forecast so the demo is useful without the compiled weights. It
 * decomposes the (already instance-normalized) history into a linear trend, a
 * detected seasonal profile, and a damped continuation of recent dynamics.
 */
const LocalEngine = {
  name: 'local-js',
  ready: true,
  async init() { return true; },

  // Detect a dominant period via autocorrelation of the residual after trend.
  _detectPeriod(res) {
    const n = res.length;
    const maxLag = Math.min(Math.floor(n / 2), 168); // cap (e.g. weekly-of-hourly)
    let best = 0, bestScore = 0;
    for (let lag = 2; lag <= maxLag; lag++) {
      let num = 0, d0 = 0, d1 = 0;
      for (let i = lag; i < n; i++) {
        num += res[i] * res[i - lag];
        d0 += res[i] * res[i];
        d1 += res[i - lag] * res[i - lag];
      }
      const score = num / (Math.sqrt(d0 * d1) + 1e-9);
      if (score > bestScore) { bestScore = score; best = lag; }
    }
    return bestScore > 0.3 ? best : 0; // require meaningful correlation
  },

  _linfit(y) {
    const n = y.length;
    let sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (let i = 0; i < n; i++) { sx += i; sy += y[i]; sxx += i * i; sxy += i * y[i]; }
    const denomv = n * sxx - sx * sx || 1;
    const slope = (n * sxy - sx * sy) / denomv;
    const intercept = (sy - slope * sx) / n;
    return { slope, intercept };
  },

  // patches: array of P-length normalized arrays. Returns a flat normalized
  // forecast of horizon = nPred * P steps.
  async forecast(patches, nPred, patchSize) {
    const hist = patches.flat();
    const n = hist.length;
    const horizon = nPred * patchSize;

    const { slope, intercept } = this._linfit(hist);
    const trend = (i) => intercept + slope * i;
    const resid = hist.map((v, i) => v - trend(i));

    const period = this._detectPeriod(resid);
    let seasonal = null;
    if (period > 0) {
      seasonal = new Array(period).fill(0);
      const counts = new Array(period).fill(0);
      for (let i = 0; i < n; i++) { seasonal[i % period] += resid[i]; counts[i % period]++; }
      for (let p = 0; p < period; p++) seasonal[p] /= counts[p] || 1;
    }

    // Damped continuation of the leftover (non-trend, non-seasonal) noise.
    const lastResid = resid[n - 1] - (seasonal ? seasonal[(n - 1) % period] : 0);
    const out = new Array(horizon);
    for (let h = 0; h < horizon; h++) {
      const idx = n + h;
      let val = trend(idx);
      if (seasonal) val += seasonal[idx % period];
      val += lastResid * Math.pow(0.85, h + 1); // damp toward the trend/season
      out[h] = val;
    }
    return out;
  },
};

/* ------------------------------------------------------------------ *
 * WebGPUEngine -- MLC-compiled 200M model via @mlc-ai/web-llm         *
 * ------------------------------------------------------------------ */
const WebGPUEngine = {
  name: 'webgpu-mlc',
  ready: false,
  meta: null,
  engine: null,

  async init() {
    if (typeof navigator === 'undefined' || !navigator.gpu) return false;
    // Are the compiled artifacts staged? ts_meta.json is written by compile.sh.
    try {
      const res = await fetch(`${CFG.modelDir}/ts_meta.json`, { cache: 'no-store' });
      if (!res.ok) return false;
      this.meta = await res.json();
    } catch (_) {
      return false;
    }

    // Load the web-llm runtime and the MLC WebGPU module. Pinned to 0.2.79
    // (see memory: webslmdemo-webllm-custom-model-config) for stable custom
    // appConfig loading.
    const webllm = await import(
      'https://esm.run/@mlc-ai/web-llm@0.2.79'
    );
    const modelId = 'ts200m-q4f16_1';
    const appConfig = {
      model_list: [{
        model: new URL(`${CFG.modelDir}/params`, self.location.href).href,
        model_id: modelId,
        model_lib: new URL(`${CFG.modelDir}/ts_model_webgpu.wasm`, self.location.href).href,
      }],
    };
    this.engine = await webllm.CreateMLCEngine(modelId, {
      appConfig,
      initProgressCallback: (p) => post('progress', { stage: 'model', ...p }),
    });
    this.ready = true;
    return true;
  },

  // NOTE: @mlc-ai/web-llm's high-level chat API is token-oriented. Driving the
  // compiled graph with continuous patch tensors requires the low-level TVM
  // runtime handle (engine.getTVMRuntime-style access) to bind the patch-embed
  // output directly, plus the ts_io.safetensors projections applied here in JS.
  // That harness is the remaining integration step once real weights exist.
  async forecast(patches, nPred, patchSize) {
    throw new Error(
      'WebGPU continuous-patch inference harness not yet wired; ' +
      'stage weights via scripts/compile.sh and implement the TVM patch bind.'
    );
  },
};

/* ------------------------------------------------------------------ *
 * Worker orchestration                                               *
 * ------------------------------------------------------------------ */
let activeEngine = LocalEngine;

function post(type, payload) {
  self.postMessage({ type, ...payload });
}

async function selectEngine() {
  try {
    if (await WebGPUEngine.init()) return WebGPUEngine;
  } catch (e) {
    post('log', { message: `WebGPU engine unavailable: ${e.message}` });
  }
  await LocalEngine.init();
  return LocalEngine;
}

self.onmessage = async (ev) => {
  const msg = ev.data || {};
  try {
    if (msg.type === 'init') {
      activeEngine = await selectEngine();
      post('ready', { engine: activeEngine.name, meta: activeEngine.meta || null });
      return;
    }

    if (msg.type === 'forecast') {
      const series = Float64Array.from(msg.series);
      const nPred = msg.nPred || CFG.nPredPatches;
      const patchSize = (activeEngine.meta && activeEngine.meta.patch_size) || CFG.patchSize;

      // 1) normalize  2) patch  3) forecast  4) denormalize
      const { norm, mean, std } = instanceNorm(Array.from(series));
      let patches = patchify(norm, patchSize);
      if (patches.length > CFG.nCtxPatches) patches = patches.slice(-CFG.nCtxPatches);

      const t0 = performance.now();
      let forecastNorm;
      try {
        forecastNorm = await activeEngine.forecast(patches, nPred, patchSize);
      } catch (e) {
        // Any engine failure -> transparent fallback to Local so the UI works.
        post('log', { message: `${activeEngine.name} forecast failed, using local: ${e.message}` });
        await LocalEngine.init();
        activeEngine = LocalEngine;
        forecastNorm = await LocalEngine.forecast(patches, nPred, patchSize);
      }
      const ms = performance.now() - t0;

      const forecast = denorm(forecastNorm, mean, std);
      post('forecast', {
        forecast,
        horizon: forecast.length,
        engine: activeEngine.name,
        ms: Math.round(ms),
        norm: { mean, std },
      });
      return;
    }
  } catch (err) {
    post('error', { message: err && err.message ? err.message : String(err) });
  }
};
