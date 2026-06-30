"""GICS sector <-> BEA industry crosswalk, and a builder that turns BEA value
added by industry into sector-level quarterly features for the panel.

The mapping is approximate by nature: GICS classifies *companies*, BEA classifies
*production*. Treat it as a starting hypothesis and tune it after you see which
industry signals actually predict revenue. Codes are BEA summary-level industry
codes — run `python -m sp_panel.industry_bea --list-industries` to see the full
list and adjust below.
"""
import pandas as pd

# GICS sector -> BEA summary industry code(s). Primary driver first.
GICS_TO_BEA = {
    "Energy":                 ["21"],            # Mining (incl. oil & gas extraction)
    "Materials":              ["31ND", "33DG"],  # chemicals (nondur) + primary/fab metals (dur)
    "Industrials":            ["33DG", "48TW", "23"],  # durable mfg, transport/warehousing, construction
    "Consumer Discretionary": ["44RT", "72"],    # retail trade, accommodation & food
    "Consumer Staples":       ["31ND", "44RT"],  # nondurable mfg (food/bev), retail
    "Health Care":            ["62"],            # health care & social assistance
    "Financials":             ["52"],            # finance & insurance
    "Real Estate":            ["53"],            # real estate & rental/leasing
    "Technology":             ["51", "54"],      # information, professional/technical svcs
    "Communication Services": ["51"],            # information (media/telecom)
}


def build_sector_features(bea_va: pd.DataFrame) -> pd.DataFrame:
    """BEA value added by industry -> sector x quarter features.

    Returns: sector, calendar_quarter, bea_va, bea_va_yoy, bea_va_qoq.
    """
    df = bea_va.copy()
    df["cq"] = pd.PeriodIndex(df["calendar_quarter"], freq="Q")
    out = []
    for sector, codes in GICS_TO_BEA.items():
        sub = df[df["bea_industry"].astype(str).isin(codes)]
        if sub.empty:
            continue
        agg = sub.groupby("cq")["value"].sum().sort_index()
        s = pd.DataFrame({
            "sector": sector,
            "calendar_quarter": agg.index.astype(str),
            "bea_va": agg.values,
            "bea_va_yoy": agg.pct_change(4).values,
            "bea_va_qoq": agg.pct_change(1).values,
        })
        out.append(s)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    from . import config
    va_path = config.DATA_DIR / "bea_value_added.parquet"
    if not va_path.exists():
        raise SystemExit("Run `python -m sp_panel.industry_bea --pull` first.")
    feat = build_sector_features(pd.read_parquet(va_path))
    out = config.DATA_DIR / "bea_sector_quarterly.parquet"
    feat.to_parquet(out, index=False)
    print(f"[crosswalk] {len(feat)} sector-quarter rows, "
          f"{feat['sector'].nunique()} sectors -> {out}")


if __name__ == "__main__":
    main()