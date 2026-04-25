#!/usr/bin/env python3
"""
Pre-fetch and cache Binance futures markets and leverage tiers data.

This avoids slow network requests during freqtrade Backtesting initialization
when exchange.reload_markets() and exchange.fill_leverage_tiers() are called.

Markets are fetched via public API (no auth required).
Leverage tiers are copied from freqtrade's bundled binance_leverage_tiers.json
(the private API requires authentication).
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt

# Where to write the combined cache
CACHE_PATH = Path("/data/freqtrade_data/_meta/exchange_cache.json")

# Where freqtrade looks for cached leverage tiers (datadir/futures/)
FREQTRADE_DATADIR = Path("/data/freqtrade_data")
LEVERAGE_TIERS_CACHE = FREQTRADE_DATADIR / "futures" / "leverage_tiers_USDT.json"

# Bundled leverage tiers from freqtrade source
FREQTRADE_ROOT = Path(__file__).resolve().parents[3]  # up from scripts/ -> rolling_backtest/ -> user_custom/ -> freqtrade/
BUNDLED_LEVERAGE_TIERS = FREQTRADE_ROOT / "freqtrade" / "exchange" / "binance_leverage_tiers.json"


def fetch_markets():
    """Fetch Binance USDM futures markets via public API (no auth needed)."""
    print("Initializing Binance USDM futures via ccxt...")
    exchange = ccxt.binanceusdm({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })

    print("Fetching markets (public API)...")
    t0 = time.time()
    markets = exchange.load_markets()
    elapsed = time.time() - t0
    print(f"  Fetched {len(markets)} markets in {elapsed:.1f}s")
    return markets


def load_bundled_leverage_tiers():
    """Load leverage tiers from freqtrade's bundled JSON file."""
    if not BUNDLED_LEVERAGE_TIERS.exists():
        print(f"  WARNING: Bundled leverage tiers not found at {BUNDLED_LEVERAGE_TIERS}")
        return {}
    
    print(f"Loading bundled leverage tiers from {BUNDLED_LEVERAGE_TIERS}...")
    with open(BUNDLED_LEVERAGE_TIERS) as f:
        tiers = json.load(f)
    print(f"  Loaded leverage tiers for {len(tiers)} symbols")
    return tiers


def save_freqtrade_leverage_cache(tiers):
    """Save leverage tiers in the format freqtrade's load_cached_leverage_tiers() expects."""
    LEVERAGE_TIERS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "data": tiers,
    }
    with open(LEVERAGE_TIERS_CACHE, "w") as f:
        json.dump(cache, f)
    size_mb = LEVERAGE_TIERS_CACHE.stat().st_size / (1024 * 1024)
    print(f"  Wrote freqtrade leverage cache: {LEVERAGE_TIERS_CACHE} ({size_mb:.1f} MB)")


def main():
    # Fetch markets
    markets = fetch_markets()

    # Load leverage tiers from bundled file
    leverage_tiers = load_bundled_leverage_tiers()

    # Save freqtrade-format leverage tiers cache
    if leverage_tiers:
        save_freqtrade_leverage_cache(leverage_tiers)

    # Save combined cache
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        "markets": markets,
        "leverage_tiers": leverage_tiers,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    print(f"\nWriting combined cache to {CACHE_PATH}...")
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, default=str)

    size_mb = CACHE_PATH.stat().st_size / (1024 * 1024)
    print(f"Done! Cache written: {size_mb:.1f} MB")
    print(f"  Markets: {len(markets)}")
    print(f"  Leverage tiers: {len(leverage_tiers)}")


if __name__ == "__main__":
    main()
