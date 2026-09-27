
# Sections downloaded via Yahoo Finance and how to transform the INI key
# into a real Yahoo ticker. `None` means "use the key as-is".
YFINANCE_SECTIONS = {"crypto", "traditional_stocks", "market_indices", "commodities"}

# Crypto tickers on Yahoo Finance need a "-USD" suffix (e.g. BTC -> BTC-USD).
CRYPTO_SUFFIX = "-USD"

# Market indices and commodities don't map 1:1 to their common short codes on
# Yahoo Finance, so we maintain an explicit lookup table.
INDEX_TICKER_MAP = {
    "SPX": "^GSPC",
    "DJI": "^DJI",
    "IXIC": "^IXIC",
    "RUT": "^RUT",
    "VIX": "^VIX",
    "FTSE": "^FTSE",
    "N225": "^N225",
    "GDAXI": "^GDAXI",
    "HSI": "^HSI",
}

COMMODITY_TICKER_MAP = {
    "XAU": "GC=F",  # Gold futures
    "XAG": "SI=F",  # Silver futures
    "CL": "CL=F",  # WTI crude futures
    "BRENT": "BZ=F",  # Brent crude futures
    "NG": "NG=F",  # Natural gas futures
    "HG": "HG=F",  # Copper futures
}

# Macro indicators are pulled from FRED. Not every common macro code has a
# clean FRED series; DXY has no reliable FRED series, so it is served via
# Yahoo Finance instead as a documented exception.
FRED_SERIES_MAP = {
    "CPI": "CPIAUCSL",
    "GDP": "GDP",
    "UNRATE": "UNRATE",
    "FEDFUNDS": "FEDFUNDS",
    "US10Y": "DGS10",
    "US2Y": "DGS2",
    "PPI": "PPIACO",
    "M2": "M2SL",
}
MACRO_YFINANCE_OVERRIDES = {
    "DXY": "DX-Y.NYB",
}