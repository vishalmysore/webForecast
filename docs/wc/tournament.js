/*
 * tournament.js -- Monte-Carlo of the actual 2026 World Cup format.
 *
 * 48 teams, 12 groups of 4. Each group is a round-robin (3/1/0 points). The top
 * two of every group plus the eight best third-placed teams advance to a 32-team
 * knockout (Round of 32 -> R16 -> QF -> SF -> Final). We sample real scorelines
 * per match (Poisson from the Elo-derived means) so group points AND goal
 * difference fall out naturally, then play the knockout with penalty-shootout
 * resolution on draws.
 *
 * Knockout seeding note: FIFA fixes the R32 bracket by group position with a
 * lookup table for which groups' thirds land where. We instead seed the 32
 * qualifiers on merit (group position, then points/GD/GF, then Elo) into a
 * standard 1-vs-32 bracket -- a faithful simplification that keeps the strong
 * sides apart until late without hard-coding the positional table.
 *
 * Pure JS. ~103 sampled matches/trial; 50k trials runs in a few seconds.
 */
'use strict';

import { sampleMatch, sampleKnockout } from './poisson.js';

export const STAGES = ['Knockout', 'Round of 16', 'Quarterfinal', 'Semifinal', 'Final', 'Champion'];

// Standard single-elim seed order for a power-of-two field.
function seedOrder(n) {
  let seeds = [1, 2];
  while (seeds.length < n) {
    const sum = seeds.length * 2 + 1;
    const next = [];
    for (const s of seeds) { next.push(s); next.push(sum - s); }
    seeds = next;
  }
  return seeds; // 1-based
}
const KO_ORDER = seedOrder(32);

function rankKey(s) {
  // higher is better: points, then goal difference, then goals for, then jitter
  return s.pts * 1e6 + (s.gf - s.ga) * 1e3 + s.gf + s.jit;
}

/*
 * groups: { A: [t0,t1,t2,t3], ... }  (team names)
 * teamIndex: Map name -> global index (for the counts arrays)
 * strengthOf: (name) => Elo for THIS trial
 */
export function simulate(groups, teamNames, trials = 50000, opts = {}) {
  const n = teamNames.length;
  const idx = new Map(teamNames.map((t, i) => [t, i]));
  const base = opts.elo || {};                 // name -> Elo (fixed baseline)
  const sample = typeof opts.sample === 'function' ? opts.sample : null;
  const strengthOf = (name) => (sample ? sample(name) : base[name]);

  const groupKeys = Object.keys(groups);
  const counts = STAGES.map(() => new Array(n).fill(0));

  for (let t = 0; t < trials; t++) {
    const elo = {};
    for (const name of teamNames) elo[name] = strengthOf(name);

    // ---- group stage ----
    const winners = [], runners = [], thirds = [];
    for (const g of groupKeys) {
      const members = groups[g];
      const st = {};
      for (const m of members) st[m] = { team: m, pts: 0, gf: 0, ga: 0, jit: Math.random() };
      for (let i = 0; i < members.length; i++) {
        for (let j = i + 1; j < members.length; j++) {
          const a = members[i], b = members[j];
          const [ga, gb] = sampleMatch(elo[a], elo[b], 0);
          st[a].gf += ga; st[a].ga += gb; st[b].gf += gb; st[b].ga += ga;
          if (ga > gb) st[a].pts += 3; else if (gb > ga) st[b].pts += 3;
          else { st[a].pts += 1; st[b].pts += 1; }
        }
      }
      const table = members.map((m) => st[m]).sort((x, y) => rankKey(y) - rankKey(x));
      winners.push({ ...table[0], tier: 0 });
      runners.push({ ...table[1], tier: 1 });
      thirds.push({ ...table[2], tier: 2 });
    }

    // ---- best 8 thirds ----
    thirds.sort((x, y) => rankKey(y) - rankKey(x));
    const bestThirds = thirds.slice(0, 8);

    // ---- seed the 32 qualifiers on merit ----
    const qualifiers = winners.concat(runners, bestThirds)
      .sort((x, y) => (x.tier - y.tier) || (rankKey(y) - rankKey(x)));
    for (const q of qualifiers) counts[0][idx.get(q.team)] += 1;  // reached Knockout

    // place seed s (1-based) -> qualifiers[s-1]
    let alive = KO_ORDER.map((seed) => qualifiers[seed - 1].team);

    // ---- knockout: 5 rounds, winners advance ----
    for (let round = 0; round < 5; round++) {
      const next = [];
      for (let i = 0; i < alive.length; i += 2) {
        const a = alive[i], b = alive[i + 1];
        const aWins = sampleKnockout(elo[a], elo[b]);
        const w = aWins ? a : b;
        next.push(w);
        counts[round + 1][idx.get(w)] += 1;   // reached next stage
      }
      alive = next;
    }
  }

  return { trials, teamNames, counts, stages: STAGES };
}

/*
 * Rows sorted by championship probability, with reach-probabilities per stage.
 * eloNow/eloProj are passed through for display.
 */
export function oddsTable(sim, meta = {}) {
  const { teamNames, counts, trials, stages } = sim;
  const rows = teamNames.map((team, i) => {
    const reach = {};
    stages.forEach((s, r) => { reach[s] = counts[r][i] / trials; });
    return {
      team,
      group: meta.groupOf?.[team] ?? '',
      eloNow: meta.eloNow?.[team],
      eloProj: meta.eloProj?.[team],
      champion: counts[counts.length - 1][i] / trials,
      reach,
    };
  });
  rows.sort((a, b) => b.champion - a.champion);
  return rows;
}
