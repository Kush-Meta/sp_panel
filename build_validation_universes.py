"""Build the two out-of-universe validation ticker lists (MODELING.md §9).

Universes: 10 companies per GICS sector (same 10 sectors as config.UNIVERSE),
plus a ranked backup pool per sector for the coverage swap pass:

  universe_smallcap600.csv   S&P SmallCap 600 members (definitionally not in
                             the S&P 500; ~$1-7B caps)
  universe_russell2000.csv   Russell 2000 members excluding S&P 500 AND S&P
                             600 (the genuinely small tier)

Russell 2000 membership comes from Vanguard's VTWO holdings API (VTWO tracks
the index; iShares' IWM download sits behind a bot wall). VTWO carries no
sector field, so sectors come from Nasdaq's screener API mapped to GICS names
(Finance->Financials, Basic Materials->Materials, Telecommunications->
Communication Services) — approximate GICS, documented in MODELING.md §9.
The S&P 600 list uses true GICS sectors from Wikipedia.

Selection is a seeded random sample among eligible names (CIK resolves in the
SEC master list), so the pick is systematic, reproducible, and documented.
Survivorship caveat: these are CURRENT constituents, so long-history names are
survivors — the same is true of the S&P 500 universe, keeping the comparison
like-for-like.

Run:  python build_validation_universes.py

After collecting data and assembling a universe's panel, run the one-shot
coverage swap pass (with SP_PANEL_DATA_DIR pointing at that universe's dir):

    SP_PANEL_DATA_DIR=data_sc600 python build_validation_universes.py \
        --swap-pass universe_smallcap600.csv

Selected tickers with too few labeled quarters are demoted to role
"dropped_lowcov" and same-sector backups promoted (role "selected"), then the
collection + assemble steps are re-run for the promoted names. One pass only —
the roles column in the CSV is the audit trail.
"""
import argparse
import io
import random

import pandas as pd
import requests

from sp_panel import config
from sp_panel.cik_map import load_cik_map, _norm

WIKI = "https://en.wikipedia.org/wiki/{}"
VTWO_API = ("https://investor.vanguard.com/investment-products/etfs/profile/api/"
            "VTWO/portfolio-holding/stock")
NASDAQ_SCREENER = ("https://api.nasdaq.com/api/screener/stocks"
                   "?tableonly=true&limit=25&download=true")
HEADERS = {"User-Agent": config.USER_AGENT}
BROWSER_HEADERS = {"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                  "AppleWebKit/537.36 Chrome/126.0"),
                   "Accept": "application/json"}
# Nasdaq screener labels -> GICS-style names used by this repo
NASDAQ_SECTOR_MAP = {"Finance": "Financials", "Basic Materials": "Materials",
                     "Telecommunications": "Communication Services"}

# source sector labels -> the repo's 10 (config.UNIVERSE / SECTOR_SERIES /
# BEA crosswalk keys). Utilities is intentionally absent: the S&P universe
# has no Utilities names, so validation universes skip it for comparability.
SECTOR_MAP = {
    "Information Technology": "Technology",
    "Technology": "Technology",
    "Communication": "Communication Services",
    "Communication Services": "Communication Services",
    "Consumer Discretionary": "Consumer Discretionary",
    "Consumer Staples": "Consumer Staples",
    "Energy": "Energy",
    "Financials": "Financials",
    "Health Care": "Health Care",
    "Industrials": "Industrials",
    "Materials": "Materials",
    "Real Estate": "Real Estate",
}
PER_SECTOR, BACKUPS, SEED = 10, 5, 0


def wiki_constituents(page):
    """(ticker, company, sector) rows from a Wikipedia index-constituents page."""
    r = requests.get(WIKI.format(page), headers=HEADERS, timeout=60)
    r.raise_for_status()
    for tbl in pd.read_html(io.StringIO(r.text)):
        cols = {c.lower().strip(): c for c in map(str, tbl.columns)}
        sym = next((cols[c] for c in cols if "symbol" in c or c == "ticker"), None)
        sec = next((cols[c] for c in cols if "gics sector" in c), None)
        name = next((cols[c] for c in cols if "company" in c or "security" in c), None)
        if sym and sec:
            out = tbl[[sym, sec] + ([name] if name else [])].copy()
            out.columns = ["ticker", "sector", "name"][: len(out.columns)]
            if "name" not in out.columns:
                out["name"] = ""
            return out.dropna(subset=["ticker"])
    raise SystemExit(f"no constituents table found on {page}")


def russell_constituents():
    """(ticker, name, sector) for Russell 2000 members: VTWO holdings joined
    to Nasdaq-screener sectors (VTWO's API carries no sector field)."""
    rows, url, guard = [], VTWO_API, 0
    while url and guard < 10:
        r = requests.get(url, headers=BROWSER_HEADERS, timeout=60)
        r.raise_for_status()
        d = r.json()
        rows += [{"ticker": h.get("ticker", "").strip(), "name": h.get("longName", "")}
                 for h in d.get("fund", {}).get("entity", [])]
        nxt = d.get("next", {}).get("href", "")
        # the API's next-href points at the internal host; keep our public base
        url = (VTWO_API + "?" + nxt.split("?", 1)[1]) if "?" in nxt else None
        guard += 1
    vtwo = pd.DataFrame(rows).query("ticker != ''").drop_duplicates("ticker")

    r = requests.get(NASDAQ_SCREENER, headers=BROWSER_HEADERS, timeout=60)
    r.raise_for_status()
    scr = pd.DataFrame(r.json()["data"]["rows"])[["symbol", "sector"]]
    scr["sector"] = scr["sector"].replace(NASDAQ_SECTOR_MAP)
    scr = scr.rename(columns={"symbol": "ticker"})

    out = vtwo.merge(scr, on="ticker", how="inner")
    print(f"[russell2000] VTWO holdings {len(vtwo)} | with screener sector {len(out)}")
    return out[["ticker", "sector", "name"]]


def pick(frame, exclude, label):
    """Seeded per-sector sample of eligible names -> selected + backup rows."""
    ciks = set(load_cik_map()["ticker"])
    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.strip()
    frame["sector"] = frame["sector"].map(lambda s: SECTOR_MAP.get(str(s).strip()))
    frame = frame.dropna(subset=["sector"])
    frame = frame[~frame["ticker"].map(_norm).isin(exclude)]
    frame = frame[frame["ticker"].map(_norm).isin(ciks)]        # must be an SEC filer
    frame = frame.drop_duplicates("ticker")

    rows, rng = [], random.Random(SEED)
    for sector in sorted(set(SECTOR_MAP.values())):
        if sector not in set(frame["sector"]):
            print(f"[{label}] WARNING: no candidates for {sector}")
            continue
        cands = sorted(frame[frame["sector"] == sector]["ticker"])
        rng.shuffle(cands)
        take = cands[: PER_SECTOR + BACKUPS]
        if len(take) < PER_SECTOR:
            print(f"[{label}] WARNING: only {len(take)} candidates in {sector}")
        names = frame.set_index("ticker")["name"]
        for i, tk in enumerate(take):
            rows.append({"ticker": tk, "sector": sector,
                         "role": "selected" if i < PER_SECTOR else "backup",
                         "name": names.get(tk, ""), "source": label})
    return pd.DataFrame(rows)


def swap_pass(csv_path, min_labeled=20):
    """Demote low-coverage selected tickers, promote same-sector backups.

    Reads the assembled panel from config.DATA_DIR (set SP_PANEL_DATA_DIR to
    the universe's data dir first). Edits the CSV in place; rerun the
    collection stages + assemble afterwards so promoted tickers get data.
    """
    panel = pd.read_parquet(config.DATA_DIR / "panel.parquet")
    labeled = (panel[panel["revenue_yoy_target"].notna()]
               .groupby("ticker").size())
    uni = pd.read_csv(csv_path)
    swaps = []
    for i, row in uni[uni["role"] == "selected"].iterrows():
        n = int(labeled.get(row["ticker"], 0))
        if n >= min_labeled:
            continue
        pool = uni[(uni["role"] == "backup") & (uni["sector"] == row["sector"])]
        if pool.empty:
            print(f"[swap] {row['ticker']} ({row['sector']}): {n} labeled rows "
                  f"but no backups left — keeping")
            continue
        j = pool.index[0]
        uni.loc[i, "role"] = "dropped_lowcov"
        uni.loc[j, "role"] = "selected"
        swaps.append((row["ticker"], n, uni.loc[j, "ticker"]))
    if swaps:
        uni.to_csv(csv_path, index=False)
        for old, n, new in swaps:
            print(f"[swap] {old} ({n} labeled rows) -> {new}")
        print(f"[swap] {len(swaps)} swaps written to {csv_path}; re-run "
              f"--financials --prices and assemble to pick them up")
    else:
        print(f"[swap] all selected tickers have >= {min_labeled} labeled rows")
    return swaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swap-pass", default=None, metavar="UNIVERSE_CSV",
                    help="coverage swap pass against the assembled panel in "
                         "config.DATA_DIR (see module docstring)")
    args = ap.parse_args()
    if args.swap_pass:
        swap_pass(args.swap_pass)
        return

    sp500 = wiki_constituents("List_of_S%26P_500_companies")
    sp600 = wiki_constituents("List_of_S%26P_600_companies")
    r2k = russell_constituents()
    print(f"constituents: S&P500 {len(sp500)} | S&P600 {len(sp600)} | Russell {len(r2k)}")

    sp500_syms = set(sp500["ticker"].map(_norm))
    sp600_syms = set(sp600["ticker"].map(_norm))

    sc600 = pick(sp600, exclude=sp500_syms, label="smallcap600")
    r2000 = pick(r2k, exclude=sp500_syms | sp600_syms, label="russell2000")

    for df, path in [(sc600, "universe_smallcap600.csv"),
                     (r2000, "universe_russell2000.csv")]:
        df.to_csv(config.ROOT / path, index=False)
        sel = df[df["role"] == "selected"]
        print(f"{path}: {len(sel)} selected + {len(df) - len(sel)} backups "
              f"across {sel['sector'].nunique()} sectors")


if __name__ == "__main__":
    main()
