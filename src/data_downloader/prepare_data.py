#!/usr/bin/env python3


"""
prepare_data.py
================
in this file we will load the raw data and prepare it for analysis
we will convert the 'price' value into logarithmic price change.
script will also report missing/broken files, and merges everything into single dataset.
"""

from __future__ import annotations

import argparse
import configparser
import csv
from dataclasses import dataclass
import sys
from pathlib import Path
import pandas as pd

from utils.logger import setup_logging


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "assets_list.ini"
DEFAULT_DATA_DIR = PROJECT_DIR / "data/raw"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data/normalized"

# Sections whose native reporting frequency is coarser than daily. Used only
# as a sensible fallback when frequency can't be inferred directly from the
# data's own date spacing.
KNOWN_MONTHLY_SECTIONS_HINT = {"CPI", "UNRATE", "FEDFUNDS", "PPI", "M2"}
KNOWN_QUARTERLY_SECTIONS_HINT = {"GDP"}


@dataclass
class AssetStatus:
    section: str
    symbol: str
    name: str
    status: str  # "ok", "missing", "failed"
    detail: str = ""
    frequency: str = ""
    rows: int = 0

def parse_config(config_path: Path) -> configparser.ConfigParser:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(config_path, encoding="utf-8")
    if not parser.sections():
        raise ValueError(f"No sections found in config file: {config_path}")
    return parser


def load_asset_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.empty:
        raise ValueError("file exists but contains no rows")
    return df


def extract_value_series(df: pd.DataFrame, symbol: str) -> pd.Series:
    """
    Reduce a raw downloaded DataFrame to a single representative series:
      - price-type data (yfinance) -> 'Close' if present, else first numeric column
      - macro data (FRED)          -> its single value column
    """
    if "Close" in df.columns:
        series = df["Close"]
    else:
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            raise ValueError("no numeric column found to extract")
        series = df[numeric_cols[0]]

    series = series.dropna()
    if series.empty:
        raise ValueError("value column is entirely empty after dropping NaNs")

    # Normalize the index: drop timezone (yfinance often returns tz-aware
    # timestamps; FRED does not), and truncate to calendar date only so the
    # two sources align cleanly when merged.
    idx = pd.DatetimeIndex(series.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    series.index = idx.normalize()
    series.name = symbol
    return series


def detect_frequency(series: pd.Series, symbol: str) -> str:
    """Best-effort frequency label, used only for reporting (not for logic)."""
    if len(series) < 2:
        return "unknown"
    inferred = pd.infer_freq(series.index)
    if inferred:
        return inferred
    median_gap_days = series.index.to_series().diff().dt.days.median()
    if median_gap_days is None or pd.isna(median_gap_days):
        return "unknown"
    if median_gap_days <= 3:
        return "daily"
    if median_gap_days <= 10:
        return "weekly"
    if median_gap_days <= 45:
        return "monthly"
    if median_gap_days <= 130:
        return "quarterly"
    return "irregular"


def collect_assets(
    config: configparser.ConfigParser, data_dir: Path, logger: logging.Logger
) -> tuple[list[AssetStatus], dict[str, pd.Series]]:
    statuses: list[AssetStatus] = []
    series_by_key: dict[str, pd.Series] = {}

    for section in config.sections():
        for symbol, name in config.items(section):
            label = f"[{section}] {symbol} ({name})"
            csv_path = data_dir / section / f"{symbol}.csv"

            if not csv_path.exists():
                logger.warning("%s: MISSING - no file at %s (needs re-download)", label, csv_path)
                statuses.append(AssetStatus(section, symbol, name, "missing", detail=f"no file at {csv_path}"))
                continue

            try:
                raw = load_asset_csv(csv_path)
                series = extract_value_series(raw, symbol)
                freq = detect_frequency(series, symbol)
                key = f"{section}__{symbol}"
                series_by_key[key] = series
                statuses.append(AssetStatus(section, symbol, name, "ok", frequency=freq, rows=len(series)))
                logger.info("%s: OK (%d rows, freq=%s)", label, len(series), freq)
            except Exception as exc:  # noqa: BLE001
                logger.error("%s: FAILED to prepare (%s) - needs fixing", label, exc)
                statuses.append(AssetStatus(section, symbol, name, "failed", detail=str(exc)))

    return statuses, series_by_key


def build_merged_datasets(series_by_key: dict[str, pd.Series], logger: logging.Logger):
    if not series_by_key:
        raise ValueError("no valid asset series available to merge")

    all_starts = [s.index.min() for s in series_by_key.values()]
    all_ends = [s.index.max() for s in series_by_key.values()]
    full_range = pd.date_range(start=min(all_starts), end=max(all_ends), freq="D")
    logger.info("Common daily calendar: %s -> %s (%d days)", full_range.min().date(), full_range.max().date(), len(full_range))

    levels = pd.DataFrame(index=full_range)
    for key, series in series_by_key.items():
        # Reindex onto the common daily calendar, then forward-fill: this
        # carries the last KNOWN value forward (e.g. a monthly CPI print
        # holds until the next print), and never fabricates future data.
        levels[key] = series.reindex(full_range).ffill()

    levels.index.name = "date"

    # Simple period-over-period percentage change on the forward-filled
    # levels. For daily assets this is a normal daily return; for
    # monthly/quarterly macro series it will mostly read 0% between prints
    # and jump on the day a new figure is carried in - which is expected and
    # should be interpreted with that in mind, not treated as a daily return.
    returns = levels.pct_change()
    returns.index.name = "date"

    return levels, returns


def write_report(statuses: list[AssetStatus], output_dir: Path) -> Path:
    report_path = output_dir / "prep_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "symbol", "name", "status", "frequency", "rows", "detail"])
        for s in statuses:
            writer.writerow([s.section, s.symbol, s.name, s.status, s.frequency, s.rows, s.detail])
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare downloaded asset data for analysis.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help=f"Path to the INI config file (default: {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help=f"Directory containing downloaded CSVs (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help=f"Directory to write prepared output into (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir / "normalized_data.log", "normalized_data")

    try:
        config = parse_config(Path(args.config))
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        return 1

    if not data_dir.exists():
        logger.error("Data directory not found: %s (run downloader.py first)", data_dir)
        return 1

    logger.info("Checking config '%s' against data in '%s'", args.config, data_dir)
    statuses, series_by_key = collect_assets(config, data_dir, logger)

    ok = [s for s in statuses if s.status == "ok"]
    missing = [s for s in statuses if s.status == "missing"]
    failed = [s for s in statuses if s.status == "failed"]

    logger.info("=" * 60)
    logger.info("Config check: %d ok, %d missing, %d failed (out of %d configured)", len(ok), len(missing), len(failed), len(statuses))
    if missing or failed:
        logger.info("Assets that need fixing:")
        for s in missing + failed:
            logger.info("  - [%s] %s (%s): %s [%s]", s.section, s.symbol, s.name, s.detail, s.status)

    report_path = write_report(statuses, output_dir)
    logger.info("Per-asset report written to: %s", report_path)

    if not series_by_key:
        logger.error("No valid data available to merge. Fix the assets listed above and re-run downloader.py.")
        return 2

    try:
        levels, returns = build_merged_datasets(series_by_key, logger)
    except Exception as exc:
        logger.error("Failed to build merged dataset: %s", exc)
        return 2

    levels_path = output_dir / "merged_levels.csv"
    returns_path = output_dir / "merged_returns.csv"
    levels.to_csv(levels_path)
    returns.to_csv(returns_path)

    logger.info("Merged levels (ffilled, daily)   -> %s (%d rows, %d columns)", levels_path, *levels.shape)
    logger.info("Merged returns (pct change)      -> %s (%d rows, %d columns)", returns_path, *returns.shape)
    logger.info("Done.")

    return 0 if not (missing or failed) else 2


if __name__ == "__main__":
    sys.exit(main())