"""Bi-monthly equity short interest from FINRA.

FINRA's data API (api.finra.org) requires free credentials: register an account at
https://developer.finra.org, create API credentials, and set them as env vars:

    export FINRA_CLIENT_ID=...
    export FINRA_CLIENT_SECRET=...

Unlike SEC/yfinance/FRED, this one won't run out-of-the-box without that signup,
so run.py treats it as optional and skips cleanly if creds are absent.

Note the data is reported per settlement date, twice a month, keyed by symbol.
This is *short interest* (open positions), distinct from daily short-sale volume.
"""
import os
import time
import requests
import pandas as pd

TOKEN_URL = "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token"
# ConsolidatedShortInterest covers exchange-listed (NYSE/Nasdaq) securities —
# the whole S&P universe. EquityShortInterest, despite the promising name, is
# OTC-market securities only and returns zero rows for listed tickers. API
# history starts ~2018 regardless of the requested start date.
DATA_URL = "https://api.finra.org/data/group/otcMarket/name/ConsolidatedShortInterest"


def _get_token():
    cid = os.environ.get("FINRA_CLIENT_ID")
    secret = os.environ.get("FINRA_CLIENT_SECRET")
    if not (cid and secret):
        return None
    resp = requests.post(TOKEN_URL, auth=(cid, secret),
                         params={"grant_type": "client_credentials"}, timeout=30)
    resp.raise_for_status()
    return resp.json()["access_token"]


def fetch_short_interest(tickers, start_date="2018-01-01") -> pd.DataFrame:
    """Pull short interest for the given tickers since start_date.

    Returns an empty frame (and prints a notice) if no FINRA creds are set.
    """
    token = _get_token()
    if token is None:
        print("[short] FINRA_CLIENT_ID/SECRET not set — skipping short interest. "
              "See module docstring to enable.")
        return pd.DataFrame()

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json",
               "Content-Type": "application/json"}
    frames = []
    # The dataset is large; filter by symbol in batches with a date floor.
    for tk in tickers:
        payload = {
            "limit": 1000,
            "compareFilters": [
                {"compareType": "GTE", "fieldName": "settlementDate", "fieldValue": start_date},
            ],
            "domainFilters": [
                {"fieldName": "symbolCode", "values": [tk.replace(".", "")]}
            ],
        }
        try:
            r = requests.post(DATA_URL, json=payload, headers=headers, timeout=60)
            if r.status_code == 200 and r.json():
                df = pd.DataFrame(r.json())
                df["ticker"] = tk
                frames.append(df)
            time.sleep(0.3)
        except requests.RequestException as e:
            print(f"[short] {tk}: {e}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out
