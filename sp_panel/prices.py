"""Stock prices, returns, volatility, and rolling beta vs benchmarks.

yfinance scrapes an undocumented Yahoo endpoint that aggressively rate-limits
(YFRateLimitError) and fails *silently* (empty frames). This module is built to
survive that:

  * one symbol at a time, no threads, with a pause + retry/backoff between calls
  * each symbol cached to disk (data/cache/px_<symbol>.parquet) so reruns skip
    what you already have and you accumulate across attempts through any block
  * Stooq fallback (a separate free source) for any equity Yahoo refuses
  * a coverage report so you can SEE exactly what came through and from where

Run `pip install -U yfinance curl_cffi pandas-datareader` first — old yfinance is
the single most common cause of "nothing comes through".

Outputs:
  prices_daily   : ticker, date, adj_close, volume, source
  benchmarks     : date, NDX, SPX
  prices_coverage.csv : per-ticker status (rows, span, source)
"""
import time
import random
import datetime as dt
import pandas as pd

from . import config


def _to_yf(ticker: str) -> str:
    return ticker.replace(".", "-")          # BRK.B -> BRK-B


def _period_to_start(period: str) -> str:
    """'2y'/'5y'/'10y' -> ISO start date (floor for the Stooq fallback)."""
    n = int("".join(c for c in period if c.isdigit()) or 2)
    yrs = n if period.endswith("y") else max(1, n // 12)
    return (dt.date.today() - dt.timedelta(days=365 * yrs + 5)).isoformat()


def _cache(symbol: str):
    return config.CACHE_DIR / f"px_{symbol}.parquet"


def _norm(df: pd.DataFrame, close_col: str, vol_col: str) -> pd.DataFrame:
    out = df.reset_index()
    date_col = "Date" if "Date" in out.columns else out.columns[0]
    out = out.rename(columns={date_col: "date", close_col: "adj_close", vol_col: "volume"})
    out["date"] = pd.to_datetime(out["date"], utc=True).dt.tz_localize(None).dt.normalize()
    return out[["date", "adj_close", "volume"]].dropna(subset=["adj_close"])


def _yf_history(symbol: str, period: str, retries: int, pause: float):
    import yfinance as yf
    for attempt in range(retries):
        try:
            df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
            if df is not None and not df.empty:
                return _norm(df, "Close", "Volume")
        except Exception as e:
            if attempt == retries - 1:
                print(f"[prices]   yahoo {symbol}: {type(e).__name__}")
        time.sleep(pause * (2 ** attempt) + random.uniform(0, 1.0))
    return None


def _stooq_history(symbol: str, start: str):
    try:
        from pandas_datareader import data as web
        df = web.DataReader(symbol, "stooq", start)   # descending OHLCV
        if df is not None and not df.empty:
            return _norm(df.sort_index(), "Close", "Volume")
    except Exception:
        pass
    return None


def _get_symbol(symbol, period, start, retries, pause, use_stooq, refresh):
    cp = _cache(symbol)
    if cp.exists() and not refresh:
        df = pd.read_parquet(cp)
        return df, df.attrs.get("source", "cache")
    df = _yf_history(symbol, period, retries, pause)
    source = "yahoo"
    if df is None and use_stooq:
        df = _stooq_history(symbol, start)
        source = "stooq"
    if df is not None:
        df["source"] = source
        df.to_parquet(cp, index=False)
        return df, source
    return None, None


def fetch_prices(universe: pd.DataFrame, period: str = None, retries: int = 3,
                 pause: float = 1.5, use_stooq: bool = True, refresh: bool = False):
    period = period or config.PRICE_PERIOD
    start = _period_to_start(period)
    tickers = universe["ticker"].tolist()
    sym_map = {tk: _to_yf(tk) for tk in tickers}

    rows, coverage = [], []
    for i, (tk, sym) in enumerate(sym_map.items(), 1):
        df, source = _get_symbol(sym, period, start, retries, pause, use_stooq, refresh)
        if df is None:
            print(f"[prices] {i}/{len(sym_map)} {tk}: NO DATA (yahoo+stooq failed)")
            coverage.append({"ticker": tk, "source": None, "rows": 0,
                             "first": None, "last": None})
            continue
        df = df.copy()
        df["ticker"] = tk
        rows.append(df[["ticker", "date", "adj_close", "volume", "source"]])
        coverage.append({"ticker": tk, "source": source, "rows": len(df),
                         "first": df["date"].min().date(), "last": df["date"].max().date()})
        print(f"[prices] {i}/{len(sym_map)} {tk}: {len(df)} rows ({source})")

    prices_long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # Benchmarks (Yahoo only; Stooq index symbols differ).
    bench_cols = {}
    for name, sym in config.BENCHMARKS.items():
        df, _ = _get_symbol(sym, period, start, retries, pause, False, refresh)
        if df is not None:
            bench_cols[name] = df.set_index("date")["adj_close"].rename(name)
            print(f"[prices] benchmark {name} ({sym}): {len(df)} rows")
        else:
            print(f"[prices] benchmark {name} ({sym}): NO DATA")
    benchmarks = (pd.concat(bench_cols.values(), axis=1).reset_index()
                  if bench_cols else pd.DataFrame())

    cov = pd.DataFrame(coverage)
    cov.to_csv(config.DATA_DIR / "prices_coverage.csv", index=False)
    n_ok = int((cov["rows"] > 0).sum()) if not cov.empty else 0
    print(f"[prices] coverage: {n_ok}/{len(tickers)} tickers have data "
          f"-> data/prices_coverage.csv")
    return prices_long, benchmarks


def compute_risk(prices_long: pd.DataFrame, benchmarks: pd.DataFrame):
    """Annualized vol + rolling beta vs each benchmark, per ticker per day."""
    if prices_long.empty or benchmarks.empty:
        return pd.DataFrame()
    import numpy as np
    px = prices_long.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    rets = px.pct_change(fill_method=None)
    bench = benchmarks.set_index("date").sort_index()
    bench_rets = bench.pct_change(fill_method=None)

    vol = rets.rolling(config.VOL_WINDOW).std() * np.sqrt(252)
    risk = vol.reset_index().melt(id_vars="date", var_name="ticker",
                                  value_name=f"vol_{config.VOL_WINDOW}")
    for bname in config.BENCHMARKS:
        if bname not in bench_rets:
            continue
        b = bench_rets[bname]
        var_b = b.rolling(config.BETA_WINDOW).var()
        beta = rets.rolling(config.BETA_WINDOW).cov(b).div(var_b, axis=0)
        col = f"beta_{bname}_{config.BETA_WINDOW}"
        risk = risk.merge(
            beta.reset_index().melt(id_vars="date", var_name="ticker", value_name=col),
            on=["date", "ticker"], how="left")
    return risk.sort_values(["ticker", "date"]).reset_index(drop=True)