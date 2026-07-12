"""Macroeconomic factors from FRED. Your independent variables; align to the
financial panel by quarter (with lags) downstream.

Two fetch paths:
  * FRED API (preferred) — used when env var FRED_API_KEY is set. Fast and
    reliable, especially for long daily series. Free key:
    https://fredaccount.stlouisfed.org/apikeys
  * Graph-CSV fallback (keyless) — hardened with a connect/read split timeout,
    a urllib3 retry adapter, and chunked reads. Works, but FRED's CSV endpoint
    can be slow for popular daily series (DGS10, VIX, oil, ...).

Either way: one series at a time, cached to disk (normalized as date,series_id),
failures skipped with a warning rather than crashing the stage.
"""
import io
import os
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd

from . import config

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API = "https://api.stlouisfed.org/fred/series/observations"
TIMEOUT = (10, 180)            # (connect, read) seconds
PAUSE = 0.5                    # gentle gap between requests

_SESSION = None


def _session():
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        retry = Retry(total=4, connect=4, read=4, backoff_factor=1.5,
                      status_forcelist=[429, 500, 502, 503, 504],
                      allowed_methods=["GET"])
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.headers.update({"User-Agent": config.USER_AGENT})
        _SESSION = s
    return _SESSION


def _normalize(df: pd.DataFrame, series_id: str) -> pd.DataFrame:
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")  # FRED "." -> NaN
    return df[["date", series_id]].dropna(subset=["date"])


def _fetch_via_api(series_id: str, start: str, api_key: str) -> pd.DataFrame:
    r = _session().get(FRED_API, timeout=TIMEOUT, params={
        "series_id": series_id, "api_key": api_key,
        "file_type": "json", "observation_start": start})
    r.raise_for_status()
    obs = r.json().get("observations", [])
    df = pd.DataFrame(obs)[["date", "value"]].rename(columns={"value": series_id})
    return _normalize(df, series_id)


def _fetch_via_csv(series_id: str, start: str) -> pd.DataFrame:
    r = _session().get(FRED_CSV, timeout=TIMEOUT, stream=True,
                       params={"id": series_id, "cosd": start})
    r.raise_for_status()
    raw = b"".join(r.iter_content(chunk_size=8192))   # chunked read resets the clock
    df = pd.read_csv(io.StringIO(raw.decode("utf-8", "replace")))
    return _normalize(df, series_id)


def _fetch_series(series_id: str, start: str, refresh: bool = False):
    # The start date is part of the cache key: changing config.FRED_START must
    # invalidate caches fetched with a shorter history, and a series that simply
    # begins later than `start` (e.g. broad USD in 2006) must NOT refetch forever.
    cache = config.CACHE_DIR / f"fred_{series_id}_{start}.csv"
    legacy = config.CACHE_DIR / f"fred_{series_id}.csv"
    if legacy.exists():
        legacy.unlink()
    if refresh and cache.exists():
        cache.unlink()
    if cache.exists():
        return _normalize(pd.read_csv(cache), series_id)

    api_key = os.environ.get("FRED_API_KEY")
    try:
        df = (_fetch_via_api(series_id, start, api_key) if api_key
              else _fetch_via_csv(series_id, start))
    except Exception as e:
        print(f"[macro] {series_id}: FAILED ({type(e).__name__}: {e}) — skipped")
        return None
    df.to_csv(cache, index=False)
    time.sleep(PAUSE)
    return df


def fetch_macro(refresh: bool = False) -> pd.DataFrame:
    """Daily/monthly-merged macro frame, one column per series. Missing series are
    absent (with a warning) rather than fatal."""
    via = "API" if os.environ.get("FRED_API_KEY") else "CSV (keyless)"
    print(f"[macro] fetching via {via}")
    frames = []
    for name, sid in config.FRED_SERIES.items():
        s = _fetch_series(sid, config.FRED_START, refresh=refresh)
        if s is None:
            continue
        frames.append(s.rename(columns={sid: name}).set_index("date"))
        print(f"[macro] {name} ({sid}): {len(s)} obs")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index().reset_index()


def to_quarterly(macro_daily: pd.DataFrame) -> pd.DataFrame:
    """Resample to quarter-end.

    Keeps last-of-quarter levels plus quarter averages and within-quarter
    standard deviations (variance was dropped: it is just std**2, pure
    redundancy that dilutes feature-importance attribution). The volatility
    columns matter most for daily financial-condition series such as WTI,
    VIX, USD and rates.
    """
    m = macro_daily.copy()
    m["date"] = pd.to_datetime(m["date"])
    m = m.set_index("date").sort_index()
    q_last = m.resample("QE").last()
    q_avg = m.resample("QE").mean().add_suffix("_qavg")
    q_std = m.resample("QE").std().add_suffix("_qstd")
    out = q_last.join(q_avg).join(q_std)

    if {"ust_10y", "ust_2y"}.issubset(out.columns):
        out["yield_curve_10y2y"] = out["ust_10y"] - out["ust_2y"]
    if {"ust_10y", "ust_3m"}.issubset(out.columns):
        out["yield_curve_10y3m"] = out["ust_10y"] - out["ust_3m"]
    if {"ust_10y_qavg", "ust_2y_qavg"}.issubset(out.columns):
        out["yield_curve_10y2y_qavg"] = out["ust_10y_qavg"] - out["ust_2y_qavg"]
    if {"ust_10y_qavg", "ust_3m_qavg"}.issubset(out.columns):
        out["yield_curve_10y3m_qavg"] = out["ust_10y_qavg"] - out["ust_3m_qavg"]
    return out.reset_index()
