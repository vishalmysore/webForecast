/*
 * timeSeriesWorker.js
 * ---------------------------------------------------------------------------
 * Off-main-thread inference for the from-scratch TimesFM 70M model
 * (FareedKhan-dev/timesfm-from-scratch, "small" tier), exported to ONNX and
 * run with onnxruntime-web on the WebGPU backend (wasm fallback).
 *
 * The ONNX graph is the model's `encode`: it takes a fixed 512-point context +
 * padding mask and returns per-patch horizon predictions in NORMALIZED space
 * plus the first-patch (mu, sigma). Everything around it — RevIN denorm, q50
 * feedback, and the autoregressive loop — is reproduced here in JS to exactly
 * match PatchedDecoder.forecast(). Point channel 5 = q50 median (TimesFM-2.5
 * default); channels 1 and 9 are q10/q90 for the uncertainty band.
 *
 * Engines (same interface): OnnxEngine (real model) with transparent fallback
 * to LocalEngine (dependency-free) when the model/runtime is unavailable.
 */

'use strict';

const CFG = {
  patchLen: 32,
  contextLen: 512,      // 16 patches
  horizonLen: 128,      // steps produced per forward
  numOutputs: 10,       // [mean, q10..q90]
  pointChannel: 5,      // q50
  q10Channel: 1,
  q90Channel: 9,
  modelDir: './model-config',
  ortVersion: '1.20.1',
};

/* ------------------------- preprocessing ------------------------- */
// Build the fixed [512] context + padding mask from an arbitrary-length series.
// Short series are LEFT-padded (mask=1); the model's first-patch RevIN skips the
// padded patches. Long series keep the most-recent 512 points.
function buildContext(series) {
  const L = CFG.contextLen;
  const x = new Float32Array(L);
  const pad = new Float32Array(L); // 1 = padded/missing
  if (series.length >= L) {
    x.set(series.slice(series.length - L));
  } else {
    const off = L - series.length;
    for (let i = 0; i < off; i++) { x[i] = 0; pad[i] = 1; }
    x.set(series, off);
  }
  return { x, pad };
}

/* ------------------------- LocalEngine (fallback) ------------------------- */
const LocalEngine = {
  name: 'local-js',
  meta: null,
  async init() { return true; },
  _linfit(y) {
    const n = y.length; let sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (let i = 0; i < n; i++) { sx += i; sy += y[i]; sxx += i * i; sxy += i * y[i]; }
    const d = n * sxx - sx * sx || 1;
    const slope = (n * sxy - sx * sy) / d;
    return { slope, intercept: (sy - slope * sx) / n };
  },
  _period(res) {
    const n = res.length, maxLag = Math.min(Math.floor(n / 2), 168);
    let best = 0, bestScore = 0;
    for (let lag = 2; lag <= maxLag; lag++) {
      let num = 0, d0 = 0, d1 = 0;
      for (let i = lag; i < n; i++) { num += res[i] * res[i - lag]; d0 += res[i] * res[i]; d1 += res[i - lag] * res[i - lag]; }
      const s = num / (Math.sqrt(d0 * d1) + 1e-9);
      if (s > bestScore) { bestScore = s; best = lag; }
    }
    return bestScore > 0.3 ? best : 0;
  },
  async forecast(series, horizon) {
    const hist = series.slice(-CFG.contextLen);
    const n = hist.length;
    const { slope, intercept } = this._linfit(hist);
    const trend = (i) => intercept + slope * i;
    const resid = hist.map((v, i) => v - trend(i));
    const period = this._period(resid);
    let seasonal = null;
    if (period > 0) {
      seasonal = new Array(period).fill(0); const c = new Array(period).fill(0);
      for (let i = 0; i < n; i++) { seasonal[i % period] += resid[i]; c[i % period]++; }
      for (let p = 0; p < period; p++) seasonal[p] /= c[p] || 1;
    }
    const lastR = resid[n - 1] - (seasonal ? seasonal[(n - 1) % period] : 0);
    const point = new Array(horizon), lo = new Array(horizon), hi = new Array(horizon);
    const sd = Math.sqrt(resid.reduce((a, b) => a + b * b, 0) / n);
    for (let h = 0; h < horizon; h++) {
      const idx = n + h;
      let v = trend(idx);
      if (seasonal) v += seasonal[idx % period];
      v += lastR * Math.pow(0.85, h + 1);
      point[h] = v; const band = sd * (1 + h / horizon);
      lo[h] = v - 1.28 * band; hi[h] = v + 1.28 * band; // ~q10/q90
    }
    return { point, lo, hi };
  },
};

/* ------------------------- OnnxEngine (real model) ------------------------- */
const OnnxEngine = {
  name: 'timesfm-70m-onnx',
  meta: null,
  session: null,
  ort: null,

  async init() {
    // Model artifacts staged?
    try {
      const r = await fetch(`${CFG.modelDir}/ts_meta.json`, { cache: 'no-store' });
      if (!r.ok) return false;
      this.meta = await r.json();
    } catch (_) { return false; }

    const ort = await import(
      `https://cdn.jsdelivr.net/npm/onnxruntime-web@${CFG.ortVersion}/dist/ort.webgpu.min.mjs`
    );
    this.ort = ort;
    ort.env.wasm.numThreads = 1; // no SharedArrayBuffer/COEP dependency
    ort.env.wasm.wasmPaths =
      `https://cdn.jsdelivr.net/npm/onnxruntime-web@${CFG.ortVersion}/dist/`;

    const hasGPU = typeof navigator !== 'undefined' && !!navigator.gpu;
    const providers = hasGPU ? ['webgpu', 'wasm'] : ['wasm'];
    const modelUrl = `${CFG.modelDir}/${this.meta.onnx}`;

    post('log', { message: `loading ONNX (${providers.join('>')}) from ${modelUrl}` });
    this.session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: providers,
      graphOptimizationLevel: 'all',
    });
    this.name = `timesfm-70m-onnx-${hasGPU ? 'webgpu' : 'wasm'}`;
    return true;
  },

  // One encode -> denormalized [128, numOutputs] for the LAST patch position.
  async _encodeLastPatch(series) {
    const { x, pad } = buildContext(series);
    const T = this.ort.Tensor;
    const feeds = {
      x: new T('float32', x, [1, CFG.contextLen]),
      padding: new T('float32', pad, [1, CFG.contextLen]),
    };
    const out = await this.session.run(feeds);
    const o = out.out;                        // [1, 16, 128, 10]
    const mu = out.mu.data[0], sigma = out.sigma.data[0];
    const N = o.dims[1], H = o.dims[2], Q = o.dims[3];
    const base = (N - 1) * H * Q;             // last patch offset
    const d = o.data;
    const rows = new Array(H);
    for (let h = 0; h < H; h++) {
      const off = base + h * Q;
      const row = new Float32Array(Q);
      for (let c = 0; c < Q; c++) row[c] = d[off + c] * sigma + mu; // denorm
      rows[h] = row;
    }
    return rows;                              // [128][10] original scale
  },

  async forecast(series, horizon) {
    const point = [], lo = [], hi = [];
    let ctx = series.slice();
    let produced = 0;
    while (produced < horizon) {
      const rows = await this._encodeLastPatch(ctx);
      for (let h = 0; h < CFG.horizonLen && produced < horizon; h++, produced++) {
        point.push(rows[h][CFG.pointChannel]);
        lo.push(rows[h][CFG.q10Channel]);
        hi.push(rows[h][CFG.q90Channel]);
        ctx.push(rows[h][CFG.pointChannel]);  // feed q50 back (AR)
      }
    }
    return { point, lo, hi };
  },
};

/* ------------------------- orchestration ------------------------- */
let active = LocalEngine;

function post(type, payload) { self.postMessage({ type, ...payload }); }

async function selectEngine() {
  try {
    if (await OnnxEngine.init()) return OnnxEngine;
  } catch (e) {
    post('log', { message: `ONNX engine unavailable: ${e.message}` });
  }
  await LocalEngine.init();
  return LocalEngine;
}

self.onmessage = async (ev) => {
  const msg = ev.data || {};
  try {
    if (msg.type === 'init') {
      active = await selectEngine();
      post('ready', { engine: active.name, meta: active.meta || null });
      return;
    }
    if (msg.type === 'forecast') {
      const series = Array.from(msg.series);
      const horizon = msg.horizon || CFG.horizonLen;
      const t0 = performance.now();
      let res;
      try {
        res = await active.forecast(series, horizon);
      } catch (e) {
        post('log', { message: `${active.name} failed, using local: ${e.message}` });
        await LocalEngine.init(); active = LocalEngine;
        res = await LocalEngine.forecast(series, horizon);
      }
      post('forecast', {
        forecast: res.point, lo: res.lo, hi: res.hi,
        horizon: res.point.length, engine: active.name,
        ms: Math.round(performance.now() - t0),
      });
      return;
    }
  } catch (err) {
    post('error', { message: err && err.message ? err.message : String(err) });
  }
};
