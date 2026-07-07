/*
 * wc.js -- World Cup 2026 champion-odds controller (UI thread, ES module).
 *
 * Pipeline:
 *   1. load elo.json (real Elo + the 12 groups, built offline from 49k matches)
 *   2. forecast every one of the 48 teams' monthly Elo with the TimesFM 70M
 *      worker (reused verbatim); 1-step q50 = tournament strength, q10/q90 = band
 *   3. Monte-Carlo the real format -- 12 groups -> best thirds -> 32-team
 *      knockout (tournament.js), optionally sampling each team's strength from
 *      its forecast band so forecast uncertainty flows into the odds
 *   4. render the group draw, the champion-odds table, and Elo forecast curves
 *
 * Forecasts are cached after the first run, so changing trials / the uncertainty
 * toggle re-simulates instantly without re-running the model.
 */
'use strict';

import { simulate, oddsTable, STAGES } from './tournament.js';

const $ = (id) => document.getElementById(id);
const FC_HORIZON = 12;
const COLORS = ['#38bdf8', '#f472b6', '#a3e635', '#fbbf24', '#c084fc',
                '#fb7185', '#34d399', '#60a5fa'];

const state = { data: null, ready: false, fc: null, groupOf: {}, chartRows: null };

/* ---------------- worker forecast client (strictly sequential) ---------------- */
const worker = new Worker('../timeSeriesWorker.js');
let pending = null;

worker.onmessage = (ev) => {
  const m = ev.data;
  switch (m.type) {
    case 'ready':
      state.ready = true;
      setStatus(`engine: ${m.engine}${m.meta ? ' (TimesFM model loaded)' : ' (on-device fallback)'} · ready`);
      $('runBtn').disabled = false;
      break;
    case 'forecast':
      if (pending) { pending.resolve(m); pending = null; }
      break;
    case 'error':
      if (pending) { pending.reject(new Error(m.message)); pending = null; }
      else setStatus(`error: ${m.message}`, true);
      break;
    case 'log':
      console.log('[worker]', m.message);
      break;
  }
};

function forecastOne(series, horizon = FC_HORIZON) {
  return new Promise((resolve, reject) => {
    pending = { resolve, reject };
    worker.postMessage({ type: 'forecast', series, horizon });
  });
}

/* ---------------- helpers ---------------- */
function setStatus(text, isError) {
  const el = $('status');
  el.textContent = text;
  el.style.color = isError ? '#f87171' : '';
}
function gauss(mean, sd) {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return mean + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
const pct = (p) => (p * 100).toFixed(p >= 0.1 ? 1 : 2) + '%';

/* ---------------- forecasting (once, cached) ---------------- */
async function forecastAll() {
  if (state.fc) return true;
  const teams = state.data.teams;
  const fc = { proj: {}, sd: {}, eloNow: {}, tail: {}, hist: {} };
  for (let i = 0; i < teams.length; i++) {
    const t = teams[i];
    setStatus(`forecasting Elo with TimesFM · ${i + 1}/${teams.length} · ${t.team}…`);
    let r;
    try { r = await forecastOne(t.series, FC_HORIZON); }
    catch (e) { setStatus(`forecast failed for ${t.team}: ${e.message}`, true); return false; }
    const proj = r.forecast[0];
    const lo = (r.lo && r.lo[0]) ?? proj, hi = (r.hi && r.hi[0]) ?? proj;
    fc.proj[t.team] = proj;
    fc.sd[t.team] = Math.max(1, (hi - lo) / 2.563);
    fc.eloNow[t.team] = t.elo;
    fc.tail[t.team] = r.forecast;
    fc.hist[t.team] = t.series;
  }
  state.fc = fc;
  return true;
}

/* ---------------- main run ---------------- */
async function run() {
  if (!state.ready || !state.data) return;
  $('runBtn').disabled = true;
  const trials = parseInt($('trials').value, 10);
  const useUncertainty = $('uncertain').checked;

  if (!(await forecastAll())) { $('runBtn').disabled = false; return; }

  const teamNames = state.data.teams.map((t) => t.team);
  const opts = {
    elo: state.fc.proj,
    sample: useUncertainty ? (name) => gauss(state.fc.proj[name], state.fc.sd[name]) : null,
  };

  setStatus(`simulating ${trials.toLocaleString()} tournaments (48 teams, real format)…`);
  await new Promise((r) => setTimeout(r, 0)); // yield so status paints (rAF throttled on hidden tabs)

  const t0 = performance.now();
  const sim = simulate(state.data.groups, teamNames, trials, opts);
  const rows = oddsTable(sim, {
    groupOf: state.groupOf, eloNow: state.fc.eloNow, eloProj: state.fc.proj,
  });
  const ms = Math.round(performance.now() - t0);

  state.chartRows = rows.slice(0, 8);
  renderTable(rows);
  renderChart(state.chartRows);
  setStatus(`done · 48 forecasts + ${trials.toLocaleString()} tournaments · sim ${ms}ms` +
            `${useUncertainty ? ' · forecast uncertainty on' : ''}`);
  $('runBtn').disabled = false;
}

/* ---------------- odds table ---------------- */
function renderTable(rows) {
  const max = rows[0]?.champion || 1;
  $('oddsBody').innerHTML = rows.map((r, i) => `
    <tr>
      <td>${i + 1}</td>
      <td class="team">${r.team}</td>
      <td class="grp">${r.group}</td>
      <td class="num">${Math.round(r.eloNow)}→${Math.round(r.eloProj)}</td>
      <td class="num">${pct(r.reach['Round of 16'])}</td>
      <td class="num">${pct(r.reach['Quarterfinal'])}</td>
      <td class="champ">${pct(r.champion)}</td>
      <td class="bar-col"><div class="bar"><span style="width:${(r.champion / max * 100).toFixed(1)}%"></span></div></td>
    </tr>`).join('');
}

/* ---------------- group draw ---------------- */
function renderGroups() {
  const g = state.data.groups;
  const now = state.fc ? state.fc.eloNow : Object.fromEntries(state.data.teams.map((t) => [t.team, t.elo]));
  $('groups').innerHTML = Object.entries(g).map(([k, members]) => `
    <div class="group">
      <div class="ghead">Group ${k}</div>
      ${members
        .map((m) => ({ m, e: now[m] ?? 0 }))
        .sort((a, b) => b.e - a.e)
        .map(({ m, e }) => `<div class="gteam"><span>${m}</span><span class="ge">${Math.round(e)}</span></div>`)
        .join('')}
    </div>`).join('');
}

/* ---------------- chart: Elo history + forecast tails (top 8) ---------------- */
function renderChart(rows) {
  const canvas = $('chart');
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (w === 0 || h === 0) { setTimeout(() => renderChart(rows), 32); return; }
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const HISTM = 48;
  const series = rows.map((r) => ({
    team: r.team,
    hist: state.fc.hist[r.team].slice(-HISTM),
    tail: state.fc.tail[r.team],
  }));

  let min = Infinity, max = -Infinity;
  for (const s of series) {
    for (const v of s.hist) { if (v < min) min = v; if (v > max) max = v; }
    for (const v of s.tail) { if (v < min) min = v; if (v > max) max = v; }
  }
  const padY = (max - min) * 0.08 || 10; min -= padY; max += padY;
  const padL = 46, padR = 12, padT = 10, padB = 20;
  const plotW = w - padL - padR, plotH = h - padT - padB;
  const total = HISTM + FC_HORIZON;
  const X = (i) => padL + (i / (total - 1)) * plotW;
  const Y = (v) => padT + (1 - (v - min) / (max - min)) * plotH;

  ctx.strokeStyle = 'rgba(148,163,184,0.15)';
  ctx.fillStyle = 'rgba(148,163,184,0.8)';
  ctx.font = '11px ui-monospace, monospace';
  for (let g = 0; g <= 4; g++) {
    const y = padT + (g / 4) * plotH;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
    ctx.fillText((max - (g / 4) * (max - min)).toFixed(0), 4, y + 3);
  }
  const bx = X(HISTM - 1);
  ctx.strokeStyle = 'rgba(56,189,248,0.3)'; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(bx, padT); ctx.lineTo(bx, h - padB); ctx.stroke();
  ctx.setLineDash([]);

  series.forEach((s, k) => {
    const color = COLORS[k % COLORS.length];
    ctx.strokeStyle = color; ctx.lineWidth = 1.4; ctx.globalAlpha = 0.9;
    ctx.beginPath();
    s.hist.forEach((v, i) => { const x = X(i), y = Y(v); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
    ctx.stroke();
    ctx.setLineDash([3, 3]); ctx.lineWidth = 1.6; ctx.globalAlpha = 1;
    ctx.beginPath();
    ctx.moveTo(X(HISTM - 1), Y(s.hist[s.hist.length - 1]));
    s.tail.forEach((v, i) => ctx.lineTo(X(HISTM + i), Y(v)));
    ctx.stroke(); ctx.setLineDash([]);
  });
  ctx.globalAlpha = 1;

  $('legend').innerHTML = series.map((s, k) =>
    `<span class="team-key"><i style="background:${COLORS[k % COLORS.length]}"></i>${s.team}</span>`).join('');
}

/* ---------------- init ---------------- */
async function init() {
  setStatus('loading Elo data…');
  try {
    const res = await fetch('./elo.json', { cache: 'no-store' });
    state.data = await res.json();
    for (const [k, members] of Object.entries(state.data.groups)) {
      for (const m of members) state.groupOf[m] = k;
    }
    renderGroups();
    setStatus(`loaded ${state.data.teams.length} teams · ${Object.keys(state.data.groups).length} groups · ${state.data.months.length} monthly Elo points · starting engine…`);
  } catch (e) {
    setStatus(`failed to load elo.json: ${e.message}`, true);
    return;
  }
  worker.postMessage({ type: 'init' });
}

$('runBtn').addEventListener('click', run);
window.addEventListener('resize', () => { if (state.chartRows) renderChart(state.chartRows); });
init();
