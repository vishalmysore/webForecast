/*
 * poisson.js -- the match layer for the World Cup forecast.
 *
 * TimesFM gives us each team's projected Elo at tournament time. This module
 * turns a pair of Elo ratings into a *scoreline distribution* using a
 * Dixon-Coles bivariate-Poisson model, then into win/draw/loss and (for
 * knockouts) an advance probability.
 *
 *   dr        = Ra - Rb + home_adv           (home_adv = 0 at a neutral WC)
 *   lambdaA   = L0 * exp(+B * dr)            expected goals, team A
 *   lambdaB   = L0 * exp(-B * dr)            expected goals, team B
 *   P(x,y)    = Pois(x;lambdaA)*Pois(y;lambdaB) * tau(x,y)   (Dixon-Coles)
 *
 * The constants are calibrated so an even neutral match is ~1.3 goals/side
 * (total ~2.6, realistic for internationals) and a 200-Elo edge yields ~+1.4
 * goal supremacy. rho slightly reweights the low-scoring cells toward draws,
 * the classic Dixon-Coles correction. ES module, no dependencies.
 */
'use strict';

export const PARAMS = {
  L0: 1.3,        // baseline expected goals per side, even match
  B: 0.0026,      // Elo-diff -> log-lambda slope (200 Elo ~ +1.4 supremacy)
  rho: -0.06,     // Dixon-Coles low-score dependence
  homeAdv: 0,     // World Cup 2026 -> all effectively neutral for the sim
  maxGoals: 8,    // truncate the scoreline grid
};

function poissonPmf(k, lambda) {
  // exp(-l) * l^k / k!
  let p = Math.exp(-lambda);
  for (let i = 1; i <= k; i++) p *= lambda / i;
  return p;
}

// Dixon-Coles tau: dependence adjustment for the four lowest scorelines.
function tau(x, y, la, lb, rho) {
  if (x === 0 && y === 0) return 1 - la * lb * rho;
  if (x === 0 && y === 1) return 1 + la * rho;
  if (x === 1 && y === 0) return 1 + lb * rho;
  if (x === 1 && y === 1) return 1 - rho;
  return 1;
}

export function eloToLambdas(rA, rB, homeAdv = PARAMS.homeAdv) {
  const dr = rA - rB + homeAdv;
  const la = PARAMS.L0 * Math.exp(PARAMS.B * dr);
  const lb = PARAMS.L0 * Math.exp(-PARAMS.B * dr);
  return { la, lb };
}

// Knuth's algorithm for a Poisson draw (fine for the small lambdas here).
export function samplePoisson(lambda) {
  const L = Math.exp(-lambda);
  let k = 0, p = 1;
  do { k++; p *= Math.random(); } while (p > L);
  return k - 1;
}

/*
 * Sample one match's scoreline from the Elo-derived Poisson means.
 * Returns [goalsA, goalsB]. Used by the tournament Monte-Carlo, where we need
 * actual scores (for group points + goal difference), not just outcome probs.
 */
export function sampleMatch(rA, rB, homeAdv = 0) {
  const { la, lb } = eloToLambdas(rA, rB, homeAdv);
  return [samplePoisson(la), samplePoisson(lb)];
}

// Sample a knockout tie: sampleMatch, then a shootout on a draw (stronger side
// slightly favoured, capped near 50/50). Returns true if A advances.
export function sampleKnockout(rA, rB) {
  const [a, b] = sampleMatch(rA, rB, 0);
  if (a > b) return true;
  if (b > a) return false;
  const shootoutA = 0.5 + Math.max(-0.12, Math.min(0.12, (rA - rB) / 4000));
  return Math.random() < shootoutA;
}

/*
 * Full scoreline distribution + aggregated outcome probabilities.
 * Returns { la, lb, pHome, pDraw, pAway, grid } where grid[x][y] = P(A x : y B).
 */
export function matchProbs(rA, rB, opts = {}) {
  const homeAdv = opts.homeAdv ?? PARAMS.homeAdv;
  const rho = opts.rho ?? PARAMS.rho;
  const M = opts.maxGoals ?? PARAMS.maxGoals;
  const { la, lb } = eloToLambdas(rA, rB, homeAdv);

  const pa = new Array(M + 1), pb = new Array(M + 1);
  for (let i = 0; i <= M; i++) { pa[i] = poissonPmf(i, la); pb[i] = poissonPmf(i, lb); }

  const grid = [];
  let pHome = 0, pDraw = 0, pAway = 0, total = 0;
  for (let x = 0; x <= M; x++) {
    grid[x] = [];
    for (let y = 0; y <= M; y++) {
      const p = pa[x] * pb[y] * tau(x, y, la, lb, rho);
      grid[x][y] = p;
      total += p;
      if (x > y) pHome += p; else if (x < y) pAway += p; else pDraw += p;
    }
  }
  // renormalize (truncation + tau lose a little mass)
  pHome /= total; pDraw /= total; pAway /= total;
  return { la, lb, pHome, pDraw, pAway, grid, total };
}

/*
 * Knockout advance probability for team A (neutral venue). A draw goes to
 * extra time / penalties, modelled as a near-coin-flip nudged by the rating
 * gap (stronger side slightly favoured, capped near 50/50).
 */
export function advanceProb(rA, rB, opts = {}) {
  const { pHome, pDraw, pAway } = matchProbs(rA, rB, { homeAdv: 0, ...opts });
  const shootoutA = 0.5 + Math.max(-0.12, Math.min(0.12, (rA - rB) / 4000));
  return pHome + pDraw * shootoutA;
}
