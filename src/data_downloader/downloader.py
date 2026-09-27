#!/usr/bin/env python3

from __future__ import annotations

import argparse
import configparser
import csv
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional
from utils.logger import setup_logging
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

# download raw data from yfinance and FRED (macro) sources, and write to CSV files

# Load ticker and source mappings from the shared project configuration.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.assets_configuration import (
    COMMODITY_TICKER_MAP,
    CRYPTO_SUFFIX,
    FRED_SERIES_MAP,
    INDEX_TICKER_MAP,
    MACRO_YFINANCE_OVERRIDES,
    YFINANCE_SECTIONS,
)

RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 2.0
REQUEST_SPACING_SECONDS = 1.0  # be polite to the data providers


@dataclass
class DownloadResult:
    section: str
    symbol: str
    name: str
    source: str
    resolved_ticker: str
    ok: bool
    rows: int = 0
    error: Optional[str] = None


@dataclass
class RunSummary:
    results: list = field(default_factory=list)

    def add(self, result: DownloadResult) -> None:
        self.results.append(result)

    @property
    def successes(self):
        return [r for r in self.results if r.ok]

    @property
    def failures(self):
        return [r for r in self.results if not r.ok]


def with_retries(fn: Callable, attempts: int, base_delay: float, logger: logging.Logger, label: str):
    """Run `fn` with exponential backoff. Raises the last exception if all attempts fail."""
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we deliberately catch broadly here
            last_exc = exc
            if attempt < attempts:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "%s: attempt %d/%d failed (%s); retrying in %.1fs",
                    label, attempt, attempts, exc, delay,
                )
                time.sleep(delay)
            else:
                logger.error("%s: all %d attempts failed (%s)", label, attempts, exc)
    raise last_exc


def parse_config(config_path: Path) -> configparser.ConfigParser:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    parser = configparser.ConfigParser()
    # Preserve key case (tickers are often uppercase and case-sensitive downstream).
    parser.optionxform = str
    parser.read(config_path, encoding="utf-8")

    if not parser.sections():
        raise ValueError(f"No sections found in config file: {config_path}")

    return parser


def resolve_ticker(section: str, symbol: str) -> tuple[str, str]:
    """
    Returns (source, resolved_ticker) for a given section/symbol pair.
    source is one of: "yfinance", "fred", "unsupported".
    """
    if section == "crypto":
        ticker = symbol if symbol.upper().endswith("-USD") else f"{symbol}{CRYPTO_SUFFIX}"
        return "yfinance", ticker

    if section == "traditional_stocks":
        return "yfinance", symbol

    if section == "market_indices":
        return "yfinance", INDEX_TICKER_MAP.get(symbol, symbol)

    if section == "commodities":
        return "yfinance", COMMODITY_TICKER_MAP.get(symbol, symbol)

    if section == "macro_indicators":
        if symbol in MACRO_YFINANCE_OVERRIDES:
            return "yfinance", MACRO_YFINANCE_OVERRIDES[symbol]
        if symbol in FRED_SERIES_MAP:
            return "fred", FRED_SERIES_MAP[symbol]
        return "unsupported", symbol

    return "unsupported", symbol


def download_yfinance(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if df is None or df.empty:
        raise ValueError("no data returned")
    return df

def download_yfinance(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if df is None or df.empty:
        raise ValueError("no data returned")

    # yfinance return MultiIndex columns (price field, ticker, date) - so flatten it
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "Date"

    return df

def download_fred(series: str, start: datetime, end: datetime) -> pd.DataFrame:
    if pdr is None:
        raise RuntimeError("pandas_datareader is not installed")
    df = pdr.DataReader(series, "fred", start, end)
    if df is None or df.empty:
        raise ValueError("no data returned")
    return df


def download_one(
        section: str,
        symbol: str,
        name: str,
        start: datetime,
        end: datetime,
        output_dir: Path,
        logger: logging.Logger,
) -> DownloadResult:
    source, ticker = resolve_ticker(section, symbol)
    label = f"[{section}] {symbol} ({name}) -> {ticker}"

    if source == "unsupported":
        logger.warning("%s: no known data source mapping, skipping", label)
        return DownloadResult(section, symbol, name, source, ticker, ok=False, error="unsupported/unmapped symbol")

    try:
        if source == "yfinance":
            df = with_retries(lambda: download_yfinance(ticker, start, end), RETRY_ATTEMPTS, RETRY_BASE_DELAY_SECONDS,
                              logger, label)
        elif source == "fred":
            df = with_retries(lambda: download_fred(ticker, start, end), RETRY_ATTEMPTS, RETRY_BASE_DELAY_SECONDS,
                              logger, label)
        else:
            raise RuntimeError(f"unknown source '{source}'")

        section_dir = output_dir / section
        section_dir.mkdir(parents=True, exist_ok=True)
        out_path = section_dir / f"{symbol}.csv"
        df.to_csv(out_path)

        logger.info("%s: OK, %d rows -> %s", label, len(df), out_path)
        return DownloadResult(section, symbol, name, source, ticker, ok=True, rows=len(df))

    except Exception as exc:  # noqa: BLE001
        logger.error("%s: FAILED (%s)", label, exc)
        return DownloadResult(section, symbol, name, source, ticker, ok=False, error=str(exc))


def write_manifest(summary: RunSummary, output_dir: Path) -> Path:
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "symbol", "name", "source", "resolved_ticker", "status", "rows", "error"])
        for r in summary.results:
            writer.writerow([
                r.section, r.symbol, r.name, r.source, r.resolved_ticker,
                "OK" if r.ok else "FAILED", r.rows, r.error or "",
            ])
    return manifest_path


# Anchor default paths to this script's own location (not the current working
# directory), so `assets.ini` / `data` are always found relative to the
# project folder regardless of where the script is invoked from.
PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "assets_list.ini"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data/raw"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download historical market data described in an INI file.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                        help=f"Path to the INI config file (default: {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help=f"Directory to write CSV files into (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--days", type=int, default=1460, help="How many days of history to fetch (default: 365)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir / "downloader.log", "downloader")

    end = datetime.now()
    start = end - timedelta(days=args.days)

    try:
        config = parse_config(Path(args.config))
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        return 1

    logger.info("Starting download run: %s -> %s (%d days)", start.date(), end.date(), args.days)
    logger.info("Config: %s | Output dir: %s", args.config, output_dir)

    summary = RunSummary()

    for section in config.sections():
        items = list(config.items(section))
        logger.info("Section '%s': %d assets", section, len(items))
        for symbol, name in items:
            result = download_one(section, symbol, name, start, end, output_dir, logger)
            summary.add(result)
            time.sleep(REQUEST_SPACING_SECONDS)

    manifest_path = write_manifest(summary, output_dir)

    logger.info("=" * 60)
    logger.info("Run complete: %d succeeded, %d failed", len(summary.successes), len(summary.failures))
    if summary.failures:
        logger.info("Failed assets:")
        for r in summary.failures:
            logger.info("  - [%s] %s (%s): %s", r.section, r.symbol, r.name, r.error)
    logger.info("Manifest written to: %s", manifest_path)

    return 0 if not summary.failures else 2


if __name__ == "__main__":
    sys.exit(main())