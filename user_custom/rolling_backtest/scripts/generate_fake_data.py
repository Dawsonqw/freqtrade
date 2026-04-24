"""Generate fake OHLCV feather files for smoke testing."""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta


def generate_ohlcv(pair_slug: str, start: str, end: str, timeframe_min: int = 5) -> pd.DataFrame:
    """Generate realistic-ish OHLCV data for a pair."""
    start_dt = datetime.strptime(start, "%Y%m%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end, "%Y%m%d").replace(tzinfo=timezone.utc)

    dates = pd.date_range(start_dt, end_dt, freq=f"{timeframe_min}min", inclusive="left")
    n = len(dates)

    # Random walk for price
    np.random.seed(hash(pair_slug) % 2**31)
    base_price = 50000 if "BTC" in pair_slug else 3000
    returns = np.random.normal(0, 0.001, n)
    close = base_price * np.exp(np.cumsum(returns))

    # Generate OHLCV from close
    noise = np.random.uniform(0.0005, 0.003, n)
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.random.uniform(100, 10000, n)

    df = pd.DataFrame({
        "date": dates[:n],
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    return df


def main():
    out_dir = Path("/tmp/fake_ft_data/futures")
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = {
        "BTC_USDT_USDT": "BTC_USDT_USDT-5m-futures.feather",
        "ETH_USDT_USDT": "ETH_USDT_USDT-5m-futures.feather",
    }

    for pair_slug, filename in pairs.items():
        df = generate_ohlcv(pair_slug, "20231201", "20240501")
        path = out_dir / filename
        df.to_feather(path)
        print(f"Generated {path}: {len(df)} candles, {df['date'].min()} ~ {df['date'].max()}")

    print(f"\nFake data dir: {out_dir}")


if __name__ == "__main__":
    main()
