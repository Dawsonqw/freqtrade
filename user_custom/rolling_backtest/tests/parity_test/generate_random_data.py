#!/usr/bin/env python3
"""Generate deterministic random OHLCV feather data for parity testing.

Produces 4 years of 5m candles for a single pair (TEST/USDT:USDT) using
a seeded random walk — fully reproducible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

PAIR = "TEST_USDT_USDT"
TIMEFRAME = "5m"
SEED = 42
START = "2022-01-01"
END = "2025-12-31 23:55:00"
OUTPUT_DIR = Path("/data/freqtrade_data/parity_test/futures")


def generate() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    # Build date index — 5m candles, UTC
    dates = pd.date_range(start=START, end=END, freq="5min", tz="UTC")
    n = len(dates)
    print(f"Generating {n} candles from {dates[0]} to {dates[-1]}")

    # Random walk for close price — start at 100, log-normal increments
    # Use larger volatility to produce more trading signals
    log_returns = rng.normal(loc=0.0, scale=0.003, size=n)  # ~0.3% per 5m
    close = 100.0 * np.exp(np.cumsum(log_returns))

    # Generate intra-candle variation
    spread = close * rng.uniform(0.001, 0.008, size=n)  # 0.1-0.8% spread
    high = close + spread * rng.uniform(0.3, 1.0, size=n)
    low = close - spread * rng.uniform(0.3, 1.0, size=n)

    # open = previous close (shifted), first candle open = close
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    # Ensure OHLC consistency: high >= max(open, close), low <= min(open, close)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    # Volume: random with some structure (higher vol = larger price moves)
    base_volume = rng.lognormal(mean=10, sigma=1.0, size=n)
    move_factor = 1.0 + np.abs(log_returns) * 100  # scale vol by price move
    volume = base_volume * move_factor

    df = pd.DataFrame({
        "date": dates,
        "open": np.round(open_, 4),
        "high": np.round(high, 4),
        "low": np.round(low, 4),
        "close": np.round(close, 4),
        "volume": np.round(volume, 2),
    })

    filename = f"{PAIR}-5m-futures.feather"
    outpath = OUTPUT_DIR / filename
    df.to_feather(outpath)
    print(f"Written {len(df)} rows to {outpath}")
    print(f"Price range: {df['close'].min():.2f} — {df['close'].max():.2f}")
    print(f"Date range: {df['date'].iloc[0]} → {df['date'].iloc[-1]}")
    return outpath


if __name__ == "__main__":
    generate()
