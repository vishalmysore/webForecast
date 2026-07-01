/*
 * main.js -- UI thread. Owns data ingest, the worker handshake, and canvas
 * charting. All tensor math happens in timeSeriesWorker.js (spec Section 4).
 */
'use strict';

const worker = new Worker('./timeSeriesWorker.js');
const $ = (id) => document.getElementById(id);

const state = {
  series: [],          // full historical series (numbers)
  forecast: [],        // last forecast (numbers)
  engine: '(starting)',
  ready: false,
};

/* ---------------- data sources ---------------- */

// Synthetic demo series: trend + daily & weekly seasonality + noise.
function demoSeries(n = 1024) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const trend = 0.02 * i;
    const daily = 8 * Math.sin((2 * Math.PI * i) / 24);
    const weekly = 4 * Math.sin((2 * Math.PI * i) / 168);
    const noise = (Math.random() - 0.5) * 2.5;
    out.push(50 + trend + daily + weekly + noise);
  }
  return out;
}

// Parse CSV/JSON: pick the last numeric column of CSV, or a JSON number array.
function parseSeries(text) {
  text = text.trim();
  if (text.startsWith('[')) {
    const arr = JSON.parse(text);
    return arr.map(Number).filter((x) => Number.isFinite(x));
  }
  const rows = text.split(/\r?\n/).filter((r) => r.trim().length);
  const nums = [];
  for (const row of rows) {
    const cells = row.split(/[,;\t]/);
    for (let c = cells.length - 1; c >= 0; c--) {
      const v = parseFloat(cells[c]);
      if (Number.isFinite(v)) { nums.push(v); break; }
    }
  }
  return nums;
}

/* ---------------- charting ---------------- */
function draw() {
  const canvas = $('chart');
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  // On first paint the canvas may not have been laid out yet (clientWidth 0).
  // Defer to the next frame rather than rendering into a 0-width bitmap.
  if (w === 0 || h === 0) { requestAnimationFrame(draw); return; }
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const hist = state.series;
  const fc = state.forecast;
  if (!hist.length) return;

  const all = hist.concat(fc);
  const min = Math.min(...all), max = Math.max(...all);
  const pad = (max - min) * 0.08 || 1;
  const lo = min - pad, hi = max + pad;
  const padL = 44, padR = 12, padT = 12, padB = 22;
  const plotW = w - padL - padR, plotH = h - padT - padB;
  const total = all.length;

  const X = (i) => padL + (i / (total - 1)) * plotW;
  const Y = (v) => padT + (1 - (v - lo) / (hi - lo)) * plotH;

  // grid + axis labels
  ctx.strokeStyle = 'rgba(148,163,184,0.18)';
  ctx.fillStyle = 'rgba(148,163,184,0.85)';
  ctx.font = '11px ui-monospace, monospace';
  ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const y = padT + (g / 4) * plotH;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
    const val = hi - (g / 4) * (hi - lo);
    ctx.fillText(val.toFixed(1), 4, y + 3);
  }

  // boundary between history and forecast
  if (fc.length) {
    const bx = X(hist.length - 1);
    ctx.strokeStyle = 'rgba(56,189,248,0.35)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(bx, padT); ctx.lineTo(bx, h - padB); ctx.stroke();
    ctx.setLineDash([]);
  }

  const line = (data, offset, color, width) => {
    if (!data.length) return;
    ctx.strokeStyle = color; ctx.lineWidth = width;
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = X(i + offset), y = Y(v);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  };

  line(hist, 0, '#94a3b8', 1.25);                          // history
  if (fc.length) {
    // connect last history point to the forecast for continuity
    line([hist[hist.length - 1], ...fc], hist.length - 1, '#38bdf8', 2);
  }
}

/* ---------------- worker wiring ---------------- */
worker.onmessage = (ev) => {
  const m = ev.data;
  switch (m.type) {
    case 'ready':
      state.ready = true;
      state.engine = m.engine;
      setStatus(`engine: ${m.engine}${m.meta ? ' (compiled model loaded)' : ' (on-device fallback)'}`);
      $('runBtn').disabled = false;
      break;
    case 'progress':
      setStatus(`loading model: ${m.text || Math.round((m.progress || 0) * 100) + '%'}`);
      break;
    case 'forecast':
      state.forecast = m.forecast;
      setStatus(`forecast: ${m.horizon} steps in ${m.ms}ms via ${m.engine}`);
      draw();
      $('runBtn').disabled = false;
      break;
    case 'log':
      console.log('[worker]', m.message);
      break;
    case 'error':
      setStatus(`error: ${m.message}`, true);
      $('runBtn').disabled = false;
      break;
  }
};

function setStatus(text, isError) {
  const el = $('status');
  el.textContent = text;
  el.style.color = isError ? '#f87171' : '';
}

function runForecast() {
  if (!state.ready || !state.series.length) return;
  $('runBtn').disabled = true;
  setStatus('forecasting...');
  const nPred = parseInt($('horizon').value, 10) || 8;
  worker.postMessage({ type: 'forecast', series: state.series, nPred });
}

/* ---------------- init ---------------- */
function loadSeries(series) {
  if (!series.length) { setStatus('no numeric data found', true); return; }
  state.series = series;
  state.forecast = [];
  setStatus(`loaded ${series.length} points`);
  draw();
}

$('demoBtn').addEventListener('click', () => loadSeries(demoSeries()));
$('runBtn').addEventListener('click', runForecast);
$('file').addEventListener('change', (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const r = new FileReader();
  r.onload = () => loadSeries(parseSeries(r.result));
  r.readAsText(f);
});
window.addEventListener('resize', draw);

loadSeries(demoSeries());
setStatus('starting inference engine...');
worker.postMessage({ type: 'init' });
