#!/usr/bin/env python3
"""
build_senti.py -- offline media-sentiment ("SentiScore") precompute for the
World Cup 2026 demo.

For every one of the 48 qualified teams we ask the free GDELT DOC 2.0 API
(https://api.gdeltproject.org, no API key) two things about the last N days of
worldwide online news coverage that mentions the team in a football context:

  1. mode=timelinetone  -> the average "tone" of that coverage. GDELT tone is a
     per-document sentiment score roughly in [-10, +10] (negative = the article
     reads negatively, positive = positively). We average it over the window.
  2. mode=artlist       -> the few most recent headlines, kept as the "why".

Tone is mapped to a 0-100 SentiScore:  score = clamp(50 + 5 * tone, 0, 100)
so neutral coverage (tone 0) -> 50, strongly positive (tone +10) -> 100,
strongly negative (tone -10) -> 0.

IMPORTANT: this is a *media-sentiment* index -- how positive/negative the public
coverage of a team is -- NOT a measurement of anyone's actual emotional state.
The UI labels it as such.

Two interchangeable sources (--source), same senti.json schema either way:

  * gdelt (--source gdelt): GDELT DOC 2.0 tone, as above. Highest quality
    (document-level tone, global) but GDELT hard-rate-limits shared/corporate/
    cloud IPs (HTTP 429) -- often unusable behind a NAT or on CI runners.
  * news  (--source news, DEFAULT): Google News RSS headlines per team, scored
    locally with VADER (compound in [-1,1]). Reachable from almost any network
    (no API, no key). tone = 10 * avg(compound) so it lands on the same ~[-10,10]
    scale, and score = clamp(50 + 5 * tone, 0, 100) is identical.

GDELT asks for <= 1 request / 5s from shared clients, so we throttle to 6s and
back off on the plain-text "Please limit requests" reply. ~48 teams => a few
minutes. Output: docs/wc/senti.json. The browser only ever sees the JSON.

Usage:
  python scripts/build_senti.py                       # news (RSS+VADER), 14d
  python scripts/build_senti.py --source gdelt        # GDELT tone
  python scripts/build_senti.py --timespan 7d --teams "Brazil,Spain"

news mode needs VADER:  python -m pip install --user vaderSentiment
"""
import argparse
import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ELO_PATH = os.path.join(HERE, "..", "docs", "wc", "elo.json")
OUT_PATH = os.path.join(HERE, "..", "docs", "wc", "senti.json")

GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
UA = "webForecast-senti/1.0 (research demo; time-series WC forecast)"
THROTTLE_S = 6.0          # GDELT wants <=1 req / 5s from shared clients
MAX_RETRIES = 4
HEADLINES = 3

# Extra disambiguating terms for names that are ambiguous or that GDELT indexes
# under a common short form. Query stays: "<team OR alias...>" (football context).
ALIASES = {
    "United States": ["USMNT", "US soccer"],
    "South Korea": ["Korea Republic"],
    "DR Congo": ["Congo DR", "Democratic Republic of Congo"],
    "Ivory Coast": ["Cote d'Ivoire"],
    "Bosnia and Herzegovina": ["Bosnia"],
    "Cape Verde": ["Cabo Verde"],
    "Czech Republic": ["Czechia"],
}


def wc_teams():
    with open(ELO_PATH, encoding="utf-8") as f:
        elo = json.load(f)
    # the 48 qualified teams are exactly the members of the 12 groups
    teams = [t for g in elo["groups"].values() for t in g]
    # keep a stable, readable order
    return sorted(set(teams))


def build_query(team):
    names = [f'"{team}"'] + [f'"{a}"' for a in ALIASES.get(team, [])]
    name_clause = "(" + " OR ".join(names) + ")" if len(names) > 1 else names[0]
    # football context so we don't pull unrelated national news
    return f'{name_clause} (soccer OR football OR "World Cup")'


def gdelt(query, mode, timespan, extra=""):
    """One throttled GDELT call with backoff on the rate-limit text reply.
    Returns parsed JSON, or None if it never came back as JSON."""
    params = {
        "query": query, "mode": mode, "timespan": timespan, "format": "json",
    }
    url = f"{GDELT}?{urllib.parse.urlencode(params)}{extra}"
    delay = THROTTLE_S
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=40) as resp:
                raw = resp.read().decode("utf-8", "replace").strip()
        except Exception as e:  # network / timeout
            raw = f"__error__ {e}"
        if raw.startswith("{"):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return None
        if not raw:
            return None  # empty body = no coverage for this query
        # rate-limit note or transient error -> back off and retry
        note = raw[:70].replace("\n", " ")
        print(f"    retry {attempt}/{MAX_RETRIES} ({note}...) waiting {delay:.0f}s",
              file=sys.stderr)
        time.sleep(delay)
        delay *= 1.8
    return None


def avg_tone(timeline_json):
    if not timeline_json:
        return None
    for s in timeline_json.get("timeline", []):
        data = s.get("data", [])
        vals = [d["value"] for d in data if isinstance(d.get("value"), (int, float))]
        if vals:
            return sum(vals) / len(vals)
    return None


def headlines(art_json):
    out = []
    for a in (art_json or {}).get("articles", [])[:HEADLINES]:
        title = (a.get("title") or "").strip()
        if not title:
            continue
        out.append({
            "title": title,
            "domain": a.get("domain", ""),
            "url": a.get("url", ""),
            "date": a.get("seendate", ""),
        })
    return out


# ---------------------------------------------------------------------------
# news source: Google News RSS headlines + local VADER sentiment
# ---------------------------------------------------------------------------
GNEWS = "https://news.google.com/rss/search"
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120 Safari/537.36")
NEWS_THROTTLE_S = 1.5          # be polite to Google News
NEWS_MAX_SCORE = 25           # headlines to average per team
_analyzer = None


def _vader():
    global _analyzer
    if _analyzer is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError:
            sys.exit("news mode needs VADER: python -m pip install --user "
                     "vaderSentiment")
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def _days(timespan):
    m = re.match(r"(\d+)\s*([dwmy])", timespan.strip().lower())
    if not m:
        return 14
    n, unit = int(m.group(1)), m.group(2)
    return n * {"d": 1, "w": 7, "m": 30, "y": 365}[unit]


def news_fetch(team, timespan):
    """Return recent Google News headlines for a team in a football context:
    [{title, domain, url, date}], newest first."""
    q = f'{build_query(team)} when:{_days(timespan)}d'
    url = (f"{GNEWS}?q={urllib.parse.quote(q)}"
           "&hl=en-US&gl=US&ceid=US:en")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": BROWSER_UA})
        with urllib.request.urlopen(req, timeout=40) as resp:
            xml = resp.read().decode("utf-8", "replace")
    except Exception as e:
        print(f"    fetch error: {e}", file=sys.stderr)
        return []
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    items = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        if not title:
            continue
        src = it.find("source")
        domain = ""
        if src is not None:
            domain = (src.get("url") or src.text or "").replace("https://", "") \
                .replace("http://", "").split("/")[0]
        # Google News titles are "Headline - Publisher"; keep the headline part
        clean = html.unescape(re.sub(r"\s+-\s+[^-]+$", "", title))
        items.append({
            "title": clean,
            "domain": domain,
            "url": (it.findtext("link") or "").strip(),
            "date": (it.findtext("pubDate") or "").strip(),
        })
    return items


def news_sentiment(team, timespan):
    """(tone in ~[-10,10], top-3 headlines) from Google News RSS + VADER."""
    items = news_fetch(team, timespan)
    if not items:
        return None, []
    an = _vader()
    comps = [an.polarity_scores(it["title"])["compound"]
             for it in items[:NEWS_MAX_SCORE]]
    comps = [c for c in comps if c is not None]
    if not comps:
        return None, items[:HEADLINES]
    tone = 10.0 * (sum(comps) / len(comps))   # onto GDELT-like [-10,10] scale
    return tone, items[:HEADLINES]


def score_from_tone(tone):
    if tone is None:
        return None
    return round(max(0.0, min(100.0, 50.0 + 5.0 * tone)))


def label_for(score):
    if score is None:
        return "no data"
    if score >= 62:
        return "buoyant"
    if score >= 54:
        return "positive"
    if score >= 46:
        return "neutral"
    if score >= 38:
        return "strained"
    return "under pressure"


def gdelt_sentiment(team, timespan):
    """(tone in ~[-10,10], top-3 headlines) from GDELT DOC 2.0."""
    q = build_query(team)
    tone_json = gdelt(q, "timelinetone", timespan)
    time.sleep(THROTTLE_S)
    art_json = gdelt(q, "artlist", timespan, extra="&maxrecords=8&sort=datedesc")
    time.sleep(THROTTLE_S)
    return avg_tone(tone_json), headlines(art_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["news", "gdelt"], default="news",
                    help="news = Google News RSS + VADER (default, works behind "
                         "most networks); gdelt = GDELT tone (rate-limits shared IPs)")
    ap.add_argument("--timespan", default="14d",
                    help="lookback window, e.g. 7d, 14d, 1m (default 14d)")
    ap.add_argument("--teams", default="",
                    help="comma-separated subset (default: all 48)")
    args = ap.parse_args()

    teams = wc_teams()
    if args.teams:
        want = {t.strip() for t in args.teams.split(",")}
        teams = [t for t in teams if t in want]

    throttle = THROTTLE_S if args.source == "gdelt" else NEWS_THROTTLE_S
    fetch = gdelt_sentiment if args.source == "gdelt" else news_sentiment
    print(f"SentiScore build: {len(teams)} teams · source={args.source} · "
          f"window {args.timespan}")
    result = {}
    for i, team in enumerate(teams, 1):
        print(f"[{i:2d}/{len(teams)}] {team}")
        tone, heads = fetch(team, args.timespan)
        if args.source == "news":
            time.sleep(throttle)
        score = score_from_tone(tone)
        result[team] = {
            "score": score,
            "tone": round(tone, 2) if tone is not None else None,
            "volume": len(heads),
            "label": label_for(score),
            "headlines": heads,
        }
        tstr = f"{tone:+.2f}" if tone is not None else "  n/a"
        print(f"        tone {tstr} -> score {score}  ({len(heads)} headlines)")

    covered = sum(1 for v in result.values() if v["score"] is not None)
    # Guard: if the source gave us nothing (e.g. an IP hard-throttled with 429s),
    # don't clobber a good senti.json — fail loudly so CI skips the commit.
    min_cov = max(1, len(teams) // 4)
    if covered < min_cov:
        print(f"\nABORT: only {covered}/{len(teams)} teams had coverage "
              f"(need >= {min_cov}); not writing {OUT_PATH}", file=sys.stderr)
        sys.exit(2)

    method = ({
        "gdelt": "GDELT DOC 2.0 average tone over %s of football coverage",
        "news": "Google News RSS headlines over %s, VADER compound sentiment",
    }[args.source] % args.timespan) + ", score=clamp(50+5*tone,0,100)"
    src = ("GDELT Project (gdeltproject.org), no API key" if args.source == "gdelt"
           else "Google News RSS + VADER (no API key)")
    out = {
        "generated": time.strftime("%Y-%m-%d"),
        "method": method,
        "source": src,
        "note": ("Media-sentiment index: how positive/negative recent public "
                 "coverage of the team is. Not a measure of anyone's emotional "
                 "state."),
        "timespan": args.timespan,
        "teams": result,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=0, ensure_ascii=False)
    kb = os.path.getsize(OUT_PATH) / 1024
    print(f"\nwrote {OUT_PATH} ({kb:.0f} KB) | {covered}/{len(teams)} teams "
          f"with sentiment coverage")


if __name__ == "__main__":
    main()
