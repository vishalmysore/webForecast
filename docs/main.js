/*
 * main.js -- UI thread. Owns data ingest, the worker handshake, and canvas
 * charting. All tensor math happens in timeSeriesWorker.js (spec Section 4).
 */
'use strict';

const worker = new Worker('./timeSeriesWorker.js');
const $ = (id) => document.getElementById(id);

const state = {
  series: [],          // full historical series (numbers)
  forecast: [],        // last forecast point (q50)
  lo: [],              // q10 band
  hi: [],              // q90 band
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

  // y-range spans history + forecast + band; x-axis spans only the plotted
  // points (history followed by the forecast), so the data fills the width.
  const yvals = hist.concat(fc, state.lo, state.hi);
  const min = Math.min(...yvals), max = Math.max(...yvals);
  const pad = (max - min) * 0.08 || 1;
  const lo = min - pad, hi = max + pad;
  const padL = 44, padR = 12, padT = 12, padB = 22;
  const plotW = w - padL - padR, plotH = h - padT - padB;
  const total = hist.length + fc.length;

  const X = (i) => padL + (i / (total - 1)) * plotW;
  const Y = (v) => padT + (1 - (v - lo) / (hi - lo)) * plotH;

  // stash the transform so the hover handler can map cursor <-> data
  state.chart = { X, Y, total, histLen: hist.length, padL, plotW, padT, plotB: h - padB, w };

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

  // uncertainty band (q10–q90) as a filled polygon
  if (state.lo.length && state.hi.length) {
    const off = hist.length - 1;
    const anchor = hist[hist.length - 1];
    ctx.fillStyle = 'rgba(56,189,248,0.14)';
    ctx.beginPath();
    ctx.moveTo(X(off), Y(anchor));
    state.hi.forEach((v, i) => ctx.lineTo(X(off + 1 + i), Y(v)));
    for (let i = state.lo.length - 1; i >= 0; i--) ctx.lineTo(X(off + 1 + i), Y(state.lo[i]));
    ctx.closePath();
    ctx.fill();
  }

  line(hist, 0, '#94a3b8', 1.25);                          // history
  if (fc.length) {
    // connect last history point to the forecast for continuity
    line([hist[hist.length - 1], ...fc], hist.length - 1, '#38bdf8', 2);
  }
}

/* ---------------- hover tooltip + crosshair ---------------- */
// Value/label at a data index (history vs forecast, with q10/q90 band).
function pointAt(idx) {
  const h = state.chart.histLen;
  if (idx <= h - 1) {
    return { type: 'history', label: `#${idx}`, value: state.series[idx] };
  }
  const fi = idx - h;
  return {
    type: 'forecast', label: `+${fi + 1}`,
    value: state.forecast[fi],
    lo: state.lo[fi], hi: state.hi[fi],
  };
}

function showHover(clientX, clientY) {
  const c = state.chart;
  const tip = $('tip');
  if (!c || !state.series.length) { tip.style.display = 'none'; return; }
  const canvas = $('chart');
  const rect = canvas.getBoundingClientRect();
  const mx = clientX - rect.left, my = clientY - rect.top;
  if (mx < c.padL || mx > c.w - 12) { tip.style.display = 'none'; draw(); return; }

  let idx = Math.round(((mx - c.padL) / c.plotW) * (c.total - 1));
  idx = Math.max(0, Math.min(c.total - 1, idx));
  const p = pointAt(idx);
  if (p.value == null || !isFinite(p.value)) { tip.style.display = 'none'; return; }

  // base chart + crosshair + marker
  draw();
  const canvasCtx = canvas.getContext('2d');
  const px = c.X(idx), py = c.Y(p.value);
  canvasCtx.save();
  canvasCtx.strokeStyle = 'rgba(148,163,184,0.5)';
  canvasCtx.lineWidth = 1;
  canvasCtx.beginPath(); canvasCtx.moveTo(px, c.padT); canvasCtx.lineTo(px, c.plotB); canvasCtx.stroke();
  const dot = p.type === 'forecast' ? '#38bdf8' : '#cbd5e1';
  canvasCtx.fillStyle = dot;
  canvasCtx.beginPath(); canvasCtx.arc(px, py, 3.5, 0, Math.PI * 2); canvasCtx.fill();
  canvasCtx.restore();

  // tooltip content
  let html = `<span class="tip-tag ${p.type}">${p.type}</span>` +
             `<b>${fmt(p.value)}</b> <span class="tip-idx">${p.label}</span>`;
  if (p.type === 'forecast' && p.lo != null && p.hi != null) {
    html += `<div class="tip-band">q10–q90: ${fmt(p.lo)} – ${fmt(p.hi)}</div>`;
  }
  tip.innerHTML = html;
  tip.style.display = 'block';
  // place near cursor, flipping left near the right edge
  const tw = tip.offsetWidth;
  let left = mx + 14; if (left + tw > c.w) left = mx - tw - 14;
  tip.style.left = `${Math.max(0, left)}px`;
  tip.style.top = `${Math.max(0, my + 14)}px`;
}

function fmt(v) {
  const a = Math.abs(v);
  if (a >= 1000) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(2);
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
      state.lo = m.lo || [];
      state.hi = m.hi || [];
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
  const horizon = parseInt($('horizon').value, 10) || 128;
  worker.postMessage({ type: 'forecast', series: state.series, horizon });
}

/* ---------------- init ---------------- */
function loadSeries(series) {
  if (!series.length) { setStatus('no numeric data found', true); return; }
  state.series = series;
  state.forecast = [];
  state.lo = [];
  state.hi = [];
  setStatus(`loaded ${series.length} points`);
  draw();
}

// Load a bundled sample file (fetch -> parse -> chart) when its name is clicked.
async function loadSampleFromUrl(url) {
  setStatus(`loading ${url.split('/').pop()}…`);
  try {
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    loadSeries(parseSeries(await res.text()));
  } catch (e) {
    setStatus(`failed to load sample: ${e.message}`, true);
  }
}
document.querySelectorAll('a[data-load]').forEach((a) => {
  a.addEventListener('click', (e) => {
    e.preventDefault();
    loadSampleFromUrl(a.getAttribute('data-load'));
  });
});

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

// hover tooltip
const chartEl = $('chart');
chartEl.addEventListener('mousemove', (e) => showHover(e.clientX, e.clientY));
chartEl.addEventListener('mouseleave', () => { $('tip').style.display = 'none'; draw(); });

loadSeries(demoSeries());
setStatus('starting inference engine...');
worker.postMessage({ type: 'init' });
