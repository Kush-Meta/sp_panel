"""Resolve tickers -> 10-digit zero-padded CIK using the SEC's official
ticker->CIK mapping (https://www.sec.gov/files/company_tickers.json).

This is the authoritative join key for every EDGAR endpoint. We normalize class
shares (BRK.B -> BRK-B) because the SEC file uses hyphen form.
"""
import json
import pandas as pd

from . import config
from .utils import sec_session, sec_get_json

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_CACHE = config.CACHE_DIR / "company_tickers.json"


def _norm(t: str) -> str:
    return t.replace(".", "-").upper().strip()


def load_cik_map(refresh: bool = False) -> pd.DataFrame:
    """Return a DataFrame: ticker, cik (int), cik_str (10-digit), title."""
    if _CACHE.exists() and not refresh:
        raw = json.loads(_CACHE.read_text())
    else:
        sess = sec_session()
        raw = sec_get_json(sess, TICKERS_URL)
        _CACHE.write_text(json.dumps(raw))

    rows = [
        {"ticker": _norm(v["ticker"]), "cik": int(v["cik_str"]),
         "cik_str": str(v["cik_str"]).zfill(10), "title": v["title"]}
        for v in raw.values()
    ]
    return pd.DataFrame(rows)


def resolve_universe(refresh: bool = False) -> pd.DataFrame:
    """Map our UNIVERSE tickers to CIKs. Flags any that don't resolve."""
    cik_map = load_cik_map(refresh=refresh).set_index("ticker")
    out = []
    for tk, sector in config.UNIVERSE.items():
        key = _norm(tk)
        if key in cik_map.index:
            row = cik_map.loc[key]
            # If a ticker maps to multiple CIKs (rare), take the first.
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            out.append({"ticker": tk, "sector": sector, "cik": int(row["cik"]),
                        "cik_str": row["cik_str"], "title": row["title"]})
        else:
            out.append({"ticker": tk, "sector": sector, "cik": None,
                        "cik_str": None, "title": None})
    df = pd.DataFrame(out)
    missing = df[df["cik"].isna()]["ticker"].tolist()
    if missing:
        print(f"[cik] WARNING: could not resolve {len(missing)}: {missing}")
    return df
