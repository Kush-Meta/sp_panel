"""BEA GDP-by-Industry puller. Needs env var BEA_API_KEY (free:
https://apps.bea.gov/API/signup/).

Pulls quarterly value added by industry — the cleanest "sector activity" signal,
essentially quarterly GDP broken out by industry. Caches raw JSON. Because the
exact TableID / Industry codes are best confirmed against the live API, two
discovery helpers print what's available:

    python -m sp_panel.industry_bea --list-tables
    python -m sp_panel.industry_bea --list-industries

Then pull (defaults to the value-added table; override with --table):

    python -m sp_panel.industry_bea --pull --table 1
"""
import os
import json
import argparse
import requests
import pandas as pd

from . import config

BEA_URL = "https://apps.bea.gov/api/data/"
DATASET = "GDPbyIndustry"
DEFAULT_TABLE = 1            # "Value Added by Industry" — confirm via --list-tables
_ROMAN = {"I": "1", "II": "2", "III": "3", "IV": "4"}


def _key():
    k = os.environ.get("BEA_API_KEY")
    if not k:
        raise SystemExit("Set BEA_API_KEY (free: https://apps.bea.gov/API/signup/).")
    return k


def _get(params):
    p = {"UserID": _key(), "ResultFormat": "json", **params}
    r = requests.get(BEA_URL, params=p, timeout=60)
    r.raise_for_status()
    js = r.json()["BEAAPI"]
    res = js.get("Results", {})
    if isinstance(res, dict) and "Error" in res:
        raise RuntimeError(f"BEA error: {res['Error']}")
    return js


def list_tables():
    js = _get({"method": "GetParameterValues", "datasetname": DATASET,
               "ParameterName": "TableID"})
    return pd.DataFrame(js["Results"]["ParamValue"])


def list_industries():
    js = _get({"method": "GetParameterValues", "datasetname": DATASET,
               "ParameterName": "Industry"})
    return pd.DataFrame(js["Results"]["ParamValue"])


def _parse(data):
    df = pd.DataFrame(data)
    df = df.rename(columns={"Industry": "bea_industry",
                            "IndustrYDescription": "description"})  # capital-Y is a BEA quirk
    df["value"] = pd.to_numeric(
        df["DataValue"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    qd = df["Quarter"].astype(str).str.strip()
    qnum = qd.map(_ROMAN)
    qnum = qnum.fillna(qd.str.extract(r"(\d)")[0])
    df["calendar_quarter"] = (df["Year"].astype(str) + "Q" + qnum)
    keep = df[["bea_industry", "description", "calendar_quarter", "value"]].copy()
    return keep.dropna(subset=["value"]).reset_index(drop=True)


def fetch_value_added(table_id=DEFAULT_TABLE, freq="Q", refresh=False):
    cache = config.CACHE_DIR / f"bea_gdpind_t{table_id}_{freq}.json"
    if cache.exists() and not refresh:
        data = json.loads(cache.read_text())
    else:
        js = _get({"method": "GetData", "datasetname": DATASET, "TableID": table_id,
                   "Industry": "ALL", "Frequency": freq, "Year": "ALL"})
        results = js["Results"]
        data = results[0]["Data"] if isinstance(results, list) else results["Data"]
        cache.write_text(json.dumps(data))
    return _parse(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-tables", action="store_true")
    ap.add_argument("--list-industries", action="store_true")
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--table", type=int, default=DEFAULT_TABLE)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    if args.list_tables:
        print(list_tables().to_string(index=False))
    if args.list_industries:
        print(list_industries().to_string(index=False))
    if args.pull:
        va = fetch_value_added(table_id=args.table, refresh=args.refresh)
        out = config.DATA_DIR / "bea_value_added.parquet"
        va.to_parquet(out, index=False)
        print(f"[bea] {len(va)} rows, {va['bea_industry'].nunique()} industries, "
              f"{va['calendar_quarter'].min()}..{va['calendar_quarter'].max()} -> {out}")


if __name__ == "__main__":
    main()