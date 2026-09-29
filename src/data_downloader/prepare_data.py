#!/usr/bin/env python3
"""
prepare_data.py
================
Load the raw per-asset CSVs (yfinance-style OHLCV for crypto/stocks/indices/
commodities, single-value FRED series for macro indicators), and prepare a
merged, daily-aligned dataset for downstream graph/GNN analysis.

For every asset we now extract, when available:
  - close   -> used for levels + log return + RSI-14
  - high/low -> used together with close for ATR-14
  - volume  -> used for a rolling volume z-score

Non-trading days (e.g. weekends for stocks, market holidays) are forward-
filled for price (the market is closed, price didn't move) but set to 0 for
volume (no trading happened - carrying the last volume forward would be a
fabricated number).

Outputs:
  merged_levels.csv   - ffilled Close, one column per asset
  merged_returns.csv  - log return of levels, one column per asset
  merged_rsi.csv      - Wilder's RSI-14 (only for assets with OHLC)
  merged_atr.csv      - ATR-14, normalized by close (only for assets with OHLC)
  merged_volume.csv   - rolling 20d z-score of volume (only for assets with volume)
  prep_report.csv     - per-asset status report (ok/missing/failed, has_ohlc)
"""

from __future__ import annotations

import argparse
import configparser
import csv
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from utils.logger import setup_logging


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "assets_list.ini"
DEFAULT_DATA_DIR = PROJECT_DIR / "data/raw"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data/normalized"

RSI_PERIOD = 14
ATR_PERIOD = 14
VOLUME_ZSCORE_WINDOW = 20


@dataclass
class AssetStatus:
    section: str
    symbol: str
    name: str
    status: str  # "ok", "missing", "failed"
    detail: str = ""
    frequency: str = ""
    rows: int = 0
    has_ohlc: bool = False


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# Extraction: now pulls Close/High/Low/Volume, not just Close
# ------------------------------------------------------------------
def extract_asset_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Reduce a raw downloaded DataFrame to a small, standardized frame:
      - price-type data (yfinance): columns close/high/low/volume (whichever exist)
        prefers 'Adj Close' for 'close' when present (splits/dividends adjusted),
        falls back to 'Close'.
      - macro data (FRED): single column, renamed to 'close' so downstream code
        (returns, etc.) can treat it uniformly. No high/low/volume.
    """
    cols = {c.lower(): c for c in df.columns}
    out = pd.DataFrame(index=df.index)

    if "close" in cols or "adj close" in cols:
        close_col = cols.get("adj close", cols.get("close"))
        out["close"] = df[close_col]
        if "high" in cols:
            out["high"] = df[cols["high"]]
        if "low" in cols:
            out["low"] = df[cols["low"]]
        if "volume" in cols:
            out["volume"] = df[cols["volume"]]
    else:
        # macro / FRED: single numeric value column
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            raise ValueError("no numeric column found to extract")
        out["close"] = df[numeric_cols[0]]

    out = out.dropna(subset=["close"])
    if out.empty:
        raise ValueError("value column is entirely empty after dropping NaNs")

    # Normalize the index: drop timezone (yfinance often returns tz-aware
    # timestamps; FRED does not), and truncate to calendar date only so
    # sources align cleanly when merged.
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    return out


def detect_frequency(series: pd.Series) -> str:
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
) -> tuple[list[AssetStatus], dict[str, pd.DataFrame]]:
    statuses: list[AssetStatus] = []
    frames_by_key: dict[str, pd.DataFrame] = {}

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
                frame = extract_asset_frame(raw, symbol)
                has_ohlc = "high" in frame.columns and "low" in frame.columns
                freq = detect_frequency(frame["close"])
                key = f"{section}__{symbol}"
                frames_by_key[key] = frame
                statuses.append(AssetStatus(section, symbol, name, "ok", frequency=freq, rows=len(frame), has_ohlc=has_ohlc))
                logger.info("%s: OK (%d rows, freq=%s, ohlc=%s)", label, len(frame), freq, has_ohlc)
            except Exception as exc:  # noqa: BLE001
                logger.error("%s: FAILED to prepare (%s) - needs fixing", label, exc)
                statuses.append(AssetStatus(section, symbol, name, "failed", detail=str(exc)))

    return statuses, frames_by_key


# ------------------------------------------------------------------
# Technical indicators (computed per-asset, on its own native calendar,
# BEFORE reindexing onto the common calendar)
# ------------------------------------------------------------------
def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing (equivalent to an EMA with alpha = 1/period)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # neutral where undefined (e.g. no losses in the window yet)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ATR_PERIOD) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    # normalize by price level so ATR is comparable across assets of very
    # different price scales (e.g. BTC at $60k vs a $5 altcoin)
    return atr / close


# ------------------------------------------------------------------
# Merge onto a common daily calendar
# ------------------------------------------------------------------
def build_merged_datasets(frames_by_key: dict[str, pd.DataFrame], logger: logging.Logger):
    if not frames_by_key:
        raise ValueError("no valid asset data available to merge")

    all_starts = [f.index.min() for f in frames_by_key.values()]
    all_ends = [f.index.max() for f in frames_by_key.values()]
    full_range = pd.date_range(start=min(all_starts), end=max(all_ends), freq="D")
    logger.info("Common daily calendar: %s -> %s (%d days)", full_range.min().date(), full_range.max().date(), len(full_range))

    levels = pd.DataFrame(index=full_range)
    returns = pd.DataFrame(index=full_range)
    rsi_df = pd.DataFrame(index=full_range)
    atr_df = pd.DataFrame(index=full_range)
    volume_df = pd.DataFrame(index=full_range)

    for key, frame in frames_by_key.items():
        has_ohlc = "high" in frame.columns and "low" in frame.columns

        # --- indicators computed on the asset's OWN calendar first ---
        rsi_native = compute_rsi(frame["close"]) if has_ohlc else None
        atr_native = compute_atr(frame["high"], frame["low"], frame["close"]) if has_ohlc else None

        # --- price: reindex onto common calendar, then ffill (market closed
        #     -> price unchanged) ---
        close_full = frame["close"].reindex(full_range).ffill()
        levels[key] = close_full
        returns[key] = np.log(close_full / close_full.shift(1))

        if has_ohlc:
            rsi_df[key] = rsi_native.reindex(full_range).ffill()
            atr_df[key] = atr_native.reindex(full_range).ffill()

        # --- volume: reindex onto common calendar, but fill non-trading
        #     days with 0 (no trading happened), never ffill ---
        if "volume" in frame.columns:
            vol_full = frame["volume"].reindex(full_range).fillna(0)
            vol_mean = vol_full.rolling(VOLUME_ZSCORE_WINDOW).mean()
            vol_std = vol_full.rolling(VOLUME_ZSCORE_WINDOW).std()
            volume_df[key] = (vol_full - vol_mean) / vol_std.replace(0, np.nan)

    for d in (levels, returns, rsi_df, atr_df, volume_df):
        d.index.name = "date"

    return levels, returns, rsi_df, atr_df, volume_df


def write_report(statuses: list[AssetStatus], output_dir: Path) -> Path:
    report_path = output_dir / "prep_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "symbol", "name", "status", "frequency", "rows", "has_ohlc", "detail"])
        for s in statuses:
            writer.writerow([s.section, s.symbol, s.name, s.status, s.frequency, s.rows, s.has_ohlc, s.detail])
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
    statuses, frames_by_key = collect_assets(config, data_dir, logger)

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

    if not frames_by_key:
        logger.error("No valid data available to merge. Fix the assets listed above and re-run downloader.py.")
        return 2

    try:
        levels, returns, rsi_df, atr_df, volume_df = build_merged_datasets(frames_by_key, logger)
    except Exception as exc:
        logger.error("Failed to build merged dataset: %s", exc)
        return 2

    outputs = {
        "merged_levels.csv": levels,
        "merged_returns.csv": returns,
        "merged_rsi.csv": rsi_df,
        "merged_atr.csv": atr_df,
        "merged_volume.csv": volume_df,
    }
    for filename, data in outputs.items():
        path = output_dir / filename
        data.to_csv(path)
        logger.info("%-20s -> %s (%d rows, %d columns)", filename, path, *data.shape)

    logger.info("Done.")
    return 0 if not (missing or failed) else 2


if __name__ == "__main__":
    sys.exit(main())