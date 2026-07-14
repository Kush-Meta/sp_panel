"""Central configuration: the universe, the XBRL concepts to pull, and settings.

The one thing you MUST change before running is CONTACT_EMAIL — the SEC requires a
real contact in the User-Agent header and will throttle/ban generic ones.

Environment overrides (for running a second universe without clobbering the
default `data/` outputs — see MODELING.md §9):
  SP_PANEL_DATA_DIR       absolute/relative path replacing ROOT/"data"
  SP_PANEL_UNIVERSE_CSV   CSV (ticker,sector[,role]) replacing the built-in
                          UNIVERSE dict; rows with role == "backup" are held
                          in reserve and not loaded
"""
import csv
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# REQUIRED: SEC asks every automated client to identify itself with a real
# contact. Put your name + email here. Requests without it get 403'd.
# ---------------------------------------------------------------------------
CONTACT_NAME = "Kush Mehta"
CONTACT_EMAIL = "kushm2004.km@gmail.com" 

USER_AGENT = f"{CONTACT_NAME} {CONTACT_EMAIL}"

# Where everything lands.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SP_PANEL_DATA_DIR", str(ROOT / "data")))
CACHE_DIR = DATA_DIR / "cache"          # raw companyfacts JSON, so reruns are free
for _d in (DATA_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# SEC fair-access guidance is 10 requests/sec. We stay well under.
SEC_MAX_RPS = 6
SEC_RETRIES = 4
SEC_BACKOFF = 2.0                       # seconds, exponential

# Price history / benchmarks for beta + volatility.
PRICE_PERIOD = "max"                     # yfinance period; gives room for trailing windows
BENCHMARKS = {"NDX": "^NDX", "SPX": "^GSPC"}
VOL_WINDOW = 252                        # trading days for annualized vol
BETA_WINDOW = 252                       # trading days for rolling beta

# ---------------------------------------------------------------------------
# The 100-company universe (ticker -> sector). Tickers match SEC convention;
# class shares like BRK.B are normalized to BRK-B at lookup time.
# ---------------------------------------------------------------------------
UNIVERSE = {
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "MCD": "Consumer Discretionary", "HD": "Consumer Discretionary",
    "LOW": "Consumer Discretionary", "F": "Consumer Discretionary",
    "GM": "Consumer Discretionary", "BKNG": "Consumer Discretionary",
    # Consumer Staples
    "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "WMT": "Consumer Staples", "COST": "Consumer Staples", "CL": "Consumer Staples",
    "KMB": "Consumer Staples", "MDLZ": "Consumer Staples", "GIS": "Consumer Staples",
    "MO": "Consumer Staples",
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Technology", "META": "Technology", "ORCL": "Technology",
    "INTC": "Technology", "CSCO": "Technology", "ADBE": "Technology",
    "CRM": "Technology",
    # Communication Services
    "NFLX": "Communication Services", "DIS": "Communication Services",
    "CMCSA": "Communication Services", "VZ": "Communication Services",
    "T": "Communication Services", "CHTR": "Communication Services",
    "TMUS": "Communication Services", "EA": "Communication Services",
    "TTWO": "Communication Services", "WBD": "Communication Services",
    # Real Estate
    "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate",
    "EQIX": "Real Estate", "PSA": "Real Estate", "O": "Real Estate",
    "SPG": "Real Estate", "DLR": "Real Estate", "AVB": "Real Estate",
    "EQR": "Real Estate",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "WMB": "Energy",
    "SLB": "Energy", "EOG": "Energy", "VLO": "Energy", "OXY": "Energy",
    "BKR": "Energy", "HAL": "Energy",
    # Materials
    "LIN": "Materials", "DOW": "Materials", "DD": "Materials", "SCCO": "Materials",
    "NEM": "Materials", "FCX": "Materials", "SHW": "Materials", "ECL": "Materials",
    "APD": "Materials", "NUE": "Materials",
    # Industrials
    "CAT": "Industrials", "GE": "Industrials", "RTX": "Industrials",
    "BA": "Industrials", "UNP": "Industrials", "DE": "Industrials",
    "ETN": "Industrials", "HON": "Industrials", "LMT": "Industrials",
    "UPS": "Industrials",
    # Health Care
    "LLY": "Health Care", "JNJ": "Health Care", "ABBV": "Health Care",
    "UNH": "Health Care", "MRK": "Health Care", "PFE": "Health Care",
    "BMY": "Health Care", "ABT": "Health Care", "AMGN": "Health Care",
    "TMO": "Health Care",
    # Financials
    "BRK.B": "Financials", "JPM": "Financials", "V": "Financials",
    "MA": "Financials", "BAC": "Financials", "MS": "Financials",
    "GS": "Financials", "C": "Financials", "USB": "Financials",
    "WFC": "Financials",
}

# Universe override for validation runs: a CSV of ticker,sector[,role] replaces
# the dict above. Sector labels must match the 10 GICS names used above (they
# key SECTOR_SERIES and the BEA crosswalk). Rows flagged role="backup" are the
# ranked reserve pool used by the coverage swap pass — not part of the universe.
_universe_csv = os.environ.get("SP_PANEL_UNIVERSE_CSV")
if _universe_csv:
    with open(_universe_csv, newline="") as _f:
        UNIVERSE = {row["ticker"].strip(): row["sector"].strip()
                    for row in csv.DictReader(_f)
                    if row.get("ticker", "").strip()
                    and row.get("role", "selected").strip() in ("selected", "")}
    if not UNIVERSE:
        raise SystemExit(f"SP_PANEL_UNIVERSE_CSV={_universe_csv} yielded no tickers")

# ---------------------------------------------------------------------------
# XBRL (us-gaap) concepts to extract from companyfacts. Revenue and a few others
# are reported under different tags across companies/eras, so we list synonyms in
# priority order and coalesce to one canonical field in edgar.py.
# ---------------------------------------------------------------------------
CONCEPTS = {
    # canonical_name : [tag synonyms in priority order]
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "RevenuesNetOfInterestExpense",
        "RevenueOtherFinancialServices",
    ],
    "net_interest_income": ["InterestIncomeExpenseNet", "InterestRevenueExpenseNet"],
    "noninterest_income": [
        "NoninterestIncome",
        "NoninterestIncomeOtherOperatingIncome",
        "NoninterestIncomeOther",
    ],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "rd_expense": ["ResearchAndDevelopmentExpense"],
    "sga_expense": ["SellingGeneralAndAdministrativeExpense"],
    "interest_expense": ["InterestExpense", "InterestExpenseDebt"],
    "tax_expense": ["IncomeTaxExpenseBenefit"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "eps_basic": ["EarningsPerShareBasic"],
    # Balance sheet (instant)
    "assets": ["Assets"],
    "assets_current": ["AssetsCurrent"],
    "liabilities": ["Liabilities"],
    "liabilities_current": ["LiabilitiesCurrent"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "inventory": ["InventoryNet"],
    "shares_outstanding": ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"],
    # Cash flow (flow)
    "cfo": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "buybacks": ["PaymentsForRepurchaseOfCommonStock"],
    "dep_amort": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
                  "DepreciationAmortizationAndAccretionNet"],
    # Accounting leading indicators (instant). Deferred revenue / contract
    # liabilities lead future sales (money collected for undelivered goods);
    # receivables growing faster than revenue is the classic Sloan-style red
    # flag; a goodwill jump is an unambiguous "acquisition closed" marker used
    # to separate inorganic from organic growth. Current portions preferred so
    # the scale matches near-term revenue.
    "deferred_revenue": ["ContractWithCustomerLiabilityCurrent", "DeferredRevenueCurrent",
                         "ContractWithCustomerLiability", "DeferredRevenue"],
    "receivables": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"],
    "goodwill": ["Goodwill"],
}

# Concepts measured at an instant (balance sheet) vs over a period (flows).
INSTANT_CONCEPTS = {
    "assets", "assets_current", "liabilities", "liabilities_current", "equity",
    "cash", "long_term_debt", "inventory", "shares_outstanding",
    "deferred_revenue", "receivables", "goodwill",
}

# ---------------------------------------------------------------------------
# FRED macro series (the independent variables). No API key needed via
# pandas_datareader's 'fred' reader.
# ---------------------------------------------------------------------------
FRED_SERIES = {
    "fed_funds": "FEDFUNDS",          # Effective federal funds rate (monthly)
    "ust_10y": "DGS10",               # 10-Year Treasury yield (daily)
    "ust_2y": "DGS2",                 # 2-Year Treasury yield (daily)
    "ust_3m": "DGS3MO",               # 3-Month Treasury (daily)
    "cpi": "CPIAUCSL",                # CPI all urban consumers (monthly)
    "core_pce": "PCEPILFE",           # Core PCE price index (monthly)
    "real_gdp": "GDPC1",              # Real GDP (quarterly)
    "unemployment": "UNRATE",         # Unemployment rate (monthly)
    "ism_pmi_proxy": "INDPRO",        # Industrial production (monthly proxy for activity)
    "retail_sales": "RSAFS",          # Advance retail sales (monthly demand)
    "pce": "PCE",                     # Personal consumption expenditures (monthly demand)
    "payrolls": "PAYEMS",             # Nonfarm payrolls (monthly labor/activity)
    "consumer_sentiment": "UMCSENT",  # University of Michigan sentiment (monthly)
    "housing_starts": "HOUST",        # Housing starts (monthly cyclicality)
    "corp_profits": "CP",             # Corporate profits after tax (quarterly)
    "usd_index": "DTWEXBGS",          # Broad USD index (daily)
    "wti_oil": "DCOILWTICO",          # WTI crude (daily)
    "vix": "VIXCLS",                  # CBOE VIX (daily)
    # Credit spread: BAA10Y instead of ICE BofA HY OAS (BAMLH0A0HYM2) because
    # the keyless fredgraph.csv endpoint caps that licensed series at ~3 years
    # of history regardless of the requested start. Baa-over-10Y tracks HY OAS
    # closely and has full daily history from 1990. (With FRED_API_KEY set, the
    # HY OAS could be restored.)
    "baa_spread": "BAA10Y",           # Moody's Baa corporate minus 10Y UST (daily)
    # --- sector-matched industry demand (feeds f_ind_* via SECTOR_SERIES, not
    # the pooled mac_* block). Stage-4 lesson: generic sector x macro
    # interactions add nothing, but a demand series matched to the sector's
    # actual product is a sharper, cross-sectionally varying instrument.
    "auto_sales": "TOTALSA",          # light vehicle sales SAAR (monthly)
    "ip_semis": "IPG3344S",           # IP: semiconductors & components (monthly)
    "ip_oil_gas": "IPG211S",          # IP: oil & gas extraction (monthly)
    "durable_orders": "DGORDER",      # durable goods new orders (monthly, 1992+)
    "ci_loans": "BUSLOANS",           # commercial & industrial loans (monthly)
    "construction_spend": "TTLCONS",  # total construction spending (monthly, 1993+)
    "pce_nondurable": "PCEND",        # PCE: nondurable goods (monthly)
    "pce_services": "PCESV",          # PCE: services (quarterly)
    "pce_health": "DHLCRC1Q027SBEA",  # PCE: health care services (quarterly)
}

# sector -> FRED_SERIES key whose growth proxies that sector's end demand.
# Housing starts (already in FRED_SERIES) covers residential; REIT-heavy Real
# Estate and Materials both lean on construction spending.
SECTOR_SERIES = {
    "Consumer Discretionary": "auto_sales",
    "Technology": "ip_semis",
    "Energy": "ip_oil_gas",
    "Industrials": "durable_orders",
    "Financials": "ci_loans",
    "Materials": "construction_spend",
    "Real Estate": "construction_spend",
    "Consumer Staples": "pce_nondurable",
    "Communication Services": "pce_services",
    "Health Care": "pce_health",
}

# Pull long macro history even though the SEC panel starts ~2011: trailing
# z-scores / cycle-state transforms need decades of context, it fills the
# 2011-2014 panel rows that were previously NaN, and FRED is free. Some series
# simply start later (broad USD 2006, HY OAS 1996) and arrive as-available.
FRED_START = "1990-01-01"
