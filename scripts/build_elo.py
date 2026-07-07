#!/usr/bin/env python3
"""
build_elo.py -- offline precompute for the World Cup 2026 forecast demo.

Reads the international-results dataset (martj42/international_results,
scripts/data/results.csv) and computes a chronological football Elo rating for
every national team using the World Football Elo Ratings method:

    R' = R + K * G * (W - We)

    We = 1 / (10^(-dr/400) + 1)          # expected result
    dr = (Ra - Rb) + home_advantage       # rating diff incl. home edge
    G  = goal-difference multiplier        # blowouts move ratings more
    K  = tournament weight                 # WC final >> friendly

It then samples each team's rating on a MONTHLY grid (last rating on/before each
month-end, carried forward) so the series is regularly spaced -- the shape a
univariate forecaster (TimesFM) expects. Output: docs/wc/elo.json.

This runs offline; the browser only ever sees the JSON.
"""
import csv
import json
import os
from collections import defaultdict
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "data", "results.csv")
OUT_PATH = os.path.join(HERE, "..", "docs", "wc", "elo.json")

BASE_RATING = 1500.0
HOME_ADV = 100.0            # World Football Elo uses 100 for the home side
MONTH_START = (2011, 1)     # first monthly sample we emit
TOP_N = 32                  # how many teams (by final Elo) to ship
SERIES_FROM = "2011-01"     # trim monthly series to keep JSON small

# Tournament weight K. Keys are matched as case-insensitive substrings against
# the dataset's `tournament` column, longest/most-specific first.
TOURNEY_K = [
    ("fifa world cup qualification", 40),
    ("fifa world cup", 60),
    ("confederations cup", 50),
    ("uefa euro qualification", 40),
    ("uefa euro", 50),
    ("copa américa", 50),
    ("copa america", 50),
    ("african cup of nations", 50),
    ("afc asian cup", 50),
    ("uefa nations league", 45),
    ("nations league", 40),
    ("gold cup", 45),
    ("qualification", 40),
    ("friendly", 20),
]
DEFAULT_K = 30


def tourney_k(name: str) -> int:
    n = name.lower()
    for key, k in TOURNEY_K:
        if key in n:
            return k
    return DEFAULT_K


def gd_multiplier(gd: int) -> float:
    """Goal-difference weighting from the World Football Elo method."""
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    if gd == 3:
        return 1.75
    return 1.75 + (gd - 3) / 8.0


def month_iter(start, end):
    (sy, sm), (ey, em) = start, end
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1


def extract_wc_groups(rows):
    """Reconstruct the 12 groups of the 2026 World Cup from the scheduled
    group-stage fixtures (each team's first three opponents form its group)."""
    from collections import defaultdict
    wc = [r for r in rows
          if r["date"].startswith("2026") and r["tournament"] == "FIFA World Cup"]
    wc.sort(key=lambda r: r["date"])
    opps = defaultdict(list)
    for r in wc:
        opps[r["home_team"]].append(r["away_team"])
        opps[r["away_team"]].append(r["home_team"])
    seen, raw = set(), []
    for team in opps:
        if team in seen:
            continue
        members = tuple(sorted({team} | set(opps[team][:3])))
        if len(members) == 4:
            for m in members:
                seen.add(m)
            raw.append(members)
    raw.sort()
    return {chr(65 + i): list(g) for i, g in enumerate(raw)}  # {'A': [...], ...}


def main():
    ratings = defaultdict(lambda: BASE_RATING)
    # history[team] = list of (date, rating_after) in chronological order
    history = defaultdict(list)
    last_match_date = {}

    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    for r in rows:
        hs, as_ = r["home_score"], r["away_score"]
        if hs in ("", "NA") or as_ in ("", "NA"):
            continue  # unplayed (incl. scheduled 2026 fixtures)
        try:
            hs, as_ = int(hs), int(as_)
        except ValueError:
            continue
        d = r["date"]
        home, away = r["home_team"], r["away_team"]
        neutral = r["neutral"].strip().upper() == "TRUE"

        Ra, Rb = ratings[home], ratings[away]
        adv = 0.0 if neutral else HOME_ADV
        dr = (Ra - Rb) + adv
        We = 1.0 / (10 ** (-dr / 400.0) + 1.0)   # expected for home
        if hs > as_:
            W = 1.0
        elif hs < as_:
            W = 0.0
        else:
            W = 0.5
        K = tourney_k(r["tournament"]) * gd_multiplier(hs - as_)
        delta = K * (W - We)
        ratings[home] = Ra + delta
        ratings[away] = Rb - delta
        history[home].append((d, ratings[home]))
        history[away].append((d, ratings[away]))
        last_match_date[home] = d
        last_match_date[away] = d

    # Monthly sampling: rating as of the last match on/before each month end.
    today = date.today()
    end = (today.year, today.month)
    months = list(month_iter(MONTH_START, end))
    month_labels = [f"{y:04d}-{m:02d}" for (y, m) in months]

    def month_end_key(y, m):
        # a string that sorts >= any date within month (y, m)
        return f"{y:04d}-{m:02d}-31"

    def monthly_series(hist):
        series, j, cur = [], 0, BASE_RATING
        for (y, m) in months:
            key = month_end_key(y, m)
            while j < len(hist) and hist[j][0] <= key:
                cur = hist[j][1]
                j += 1
            series.append(round(cur, 2))
        return series

    # The 48 qualified teams and their groups drive the tournament sim; we emit
    # an Elo series for every one of them (plus the overall top teams for context).
    groups = extract_wc_groups(rows)
    wc_teams = [t for g in groups.values() for t in g]

    monthly = {}
    for team, hist in history.items():
        if len(hist) < 30 and team not in wc_teams:
            continue
        monthly[team] = monthly_series(hist)
    # ensure every WC team has a series even with sparse history
    for team in wc_teams:
        if team not in monthly and team in history:
            monthly[team] = monthly_series(history[team])

    # trim series to SERIES_FROM
    start_idx = month_labels.index(SERIES_FROM) if SERIES_FROM in month_labels else 0
    labels = month_labels[start_idx:]

    # Emit the 48 WC teams first (sim needs all), then fill to TOP_N with the
    # strongest non-WC sides for the standalone chart/context.
    ranked = sorted(monthly.items(), key=lambda kv: kv[1][-1], reverse=True)
    ordered = [t for t in wc_teams if t in monthly]
    for name, _ in ranked:
        if name not in ordered and len(ordered) < max(TOP_N, len(wc_teams)):
            ordered.append(name)

    teams_out = []
    for name in ordered:
        series = monthly[name]
        teams_out.append({
            "team": name,
            "elo": round(series[-1], 1),
            "series": series[start_idx:],
            "last_match": last_match_date.get(name),
        })

    out = {
        "generated": today.isoformat(),
        "method": "World Football Elo (home adv 100, GD-weighted, tournament K)",
        "months": labels,
        "groups": groups,
        "teams": teams_out,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=0)
    kb = os.path.getsize(OUT_PATH) / 1024
    print(f"wrote {OUT_PATH}  ({kb:.0f} KB)")
    print(f"{len(labels)} monthly points | {len(teams_out)} teams | {len(groups)} groups")
    missing = [t for t in wc_teams if t not in monthly]
    if missing:
        print("  WARN missing Elo for:", missing)
    elo_by = {t["team"]: t["elo"] for t in teams_out}
    for g, members in groups.items():
        line = "  ".join(f"{m}({elo_by.get(m, 0):.0f})" for m in members)
        print(f"  {g}: {line}")


if __name__ == "__main__":
    main()
