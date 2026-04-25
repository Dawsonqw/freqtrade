#!/usr/bin/env python3
"""Deep diagnosis: compare indicator values between native and rolling at window boundaries."""
import json, sys, pandas as pd, numpy as np
sys.path.insert(0, '/root/workspace/freqtrade')

from freqtrade.configuration import Configuration
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.configuration import TimeRange
from freqtrade.data import history
from freqtrade.data.converter import trim_dataframes

# Load config
config = Configuration.from_files([
    'user_custom/rolling_backtest/tests/parity_test/config_parity.json'
])
config['strategy'] = 'ParityTestStrategy'
config['timerange'] = '20240101-20240301'
config['verbosity'] = 0
config['tradable_balance_ratio'] = 1.0
config['available_capital'] = config.get('dry_run_wallet', 10000.0)

bt = Backtesting(config)
strategy = bt.strategylist[0]

# Load all data
full_tr = bt.timerange
data = history.load_data(
    datadir=config['datadir'],
    pairs=['BTC/USDT:USDT'],
    timeframe='5m',
    timerange=full_tr,
    startup_candles=bt.required_startup,
    fail_without_data=False,
    data_format=config['dataformat_ohlcv'],
    candle_type=config.get('candle_type_def'),
)

pair = 'BTC/USDT:USDT'
df_full = data[pair].copy()
print(f'Full data: {len(df_full)} candles, {df_full["date"].min()} to {df_full["date"].max()}')
print(f'Startup candles: {bt.required_startup}')

# Compute indicators on full data (what native does)
df_native = strategy.advise_indicators(df_full.copy(), {'pair': pair})
df_native = strategy.advise_entry(df_native, {'pair': pair})
df_native = strategy.advise_exit(df_native, {'pair': pair})

# Trim to just the timerange (after startup)
trimmed_native = trim_dataframes({pair: df_native}, full_tr, bt.required_startup)
df_native_trimmed = trimmed_native[pair]

print(f'Native after trim: {len(df_native_trimmed)} candles')
print(f'  first: {df_native_trimmed["date"].iloc[0]}')
print(f'  last: {df_native_trimmed["date"].iloc[-1]}')

# Now simulate what rolling does: window 1 = Jan 1-31, window 2 = Jan 31-Mar 1
w1_start = pd.Timestamp('2024-01-01', tz='UTC')
w1_end = pd.Timestamp('2024-01-31', tz='UTC')
w2_start = pd.Timestamp('2024-01-31', tz='UTC')
w2_end = pd.Timestamp('2024-03-01', tz='UTC')

# Window 1: need startup candles before Jan 1
from freqtrade.exchange import timeframe_to_seconds
startup_secs = bt.required_startup * timeframe_to_seconds('5m')

w1_data_start = w1_start - pd.Timedelta(seconds=startup_secs)
mask1 = (df_full['date'] >= w1_data_start) & (df_full['date'] <= w1_end)
df_w1_raw = df_full[mask1].copy()

# Compute indicators on window 1 data
df_w1 = strategy.advise_indicators(df_w1_raw.copy(), {'pair': pair})
df_w1 = strategy.advise_entry(df_w1, {'pair': pair})
df_w1 = strategy.advise_exit(df_w1, {'pair': pair})

# Trim to window 1 (remove startup)
w1_tr = TimeRange('date', 'date', int(w1_start.timestamp()), int(w1_end.timestamp()))
trimmed_w1 = trim_dataframes({pair: df_w1}, w1_tr, bt.required_startup)
df_w1_trimmed = trimmed_w1[pair]

# Window 2: data from w2_start with startup  
w2_data_start = w2_start - pd.Timedelta(seconds=startup_secs)
mask2 = (df_full['date'] >= w2_data_start) & (df_full['date'] <= w2_end)
df_w2_raw = df_full[mask2].copy()

df_w2 = strategy.advise_indicators(df_w2_raw.copy(), {'pair': pair})
df_w2 = strategy.advise_entry(df_w2, {'pair': pair})
df_w2 = strategy.advise_exit(df_w2, {'pair': pair})

w2_tr = TimeRange('date', 'date', int(w2_start.timestamp()), int(w2_end.timestamp()))
trimmed_w2 = trim_dataframes({pair: df_w2}, w2_tr, bt.required_startup)
df_w2_trimmed = trimmed_w2[pair]

print(f'\nWindow 1 trimmed: {len(df_w1_trimmed)} candles, {df_w1_trimmed["date"].iloc[0]} to {df_w1_trimmed["date"].iloc[-1]}')
print(f'Window 2 trimmed: {len(df_w2_trimmed)} candles, {df_w2_trimmed["date"].iloc[0]} to {df_w2_trimmed["date"].iloc[-1]}')

# Check overlap
w1_dates = set(df_w1_trimmed['date'])
w2_dates = set(df_w2_trimmed['date'])
overlap = w1_dates & w2_dates
print(f'Date overlap between windows: {len(overlap)} candles')

# Compare signals at overlap points
if overlap:
    overlap_sorted = sorted(overlap)
    print(f'  Overlap range: {overlap_sorted[0]} to {overlap_sorted[-1]}')
    for dt in overlap_sorted[:5]:
        r1 = df_w1_trimmed[df_w1_trimmed['date'] == dt].iloc[0]
        r2 = df_w2_trimmed[df_w2_trimmed['date'] == dt].iloc[0]
        rn = df_native_trimmed[df_native_trimmed['date'] == dt]
        if len(rn) > 0:
            rn = rn.iloc[0]
            print(f'  {dt}:')
            print(f'    Native:  ema_f={rn["ema_fast"]:.2f} ema_s={rn["ema_slow"]:.2f} rsi={rn["rsi"]:.2f} enter_long={rn.get("enter_long",0)} enter_short={rn.get("enter_short",0)}')
            print(f'    W1:      ema_f={r1["ema_fast"]:.2f} ema_s={r1["ema_slow"]:.2f} rsi={r1["rsi"]:.2f} enter_long={r1.get("enter_long",0)} enter_short={r1.get("enter_short",0)}')
            print(f'    W2:      ema_f={r2["ema_fast"]:.2f} ema_s={r2["ema_slow"]:.2f} rsi={r2["rsi"]:.2f} enter_long={r2.get("enter_long",0)} enter_short={r2.get("enter_short",0)}')

# Compare key indicator at a specific point: right at window boundary
boundary_dt = pd.Timestamp('2024-01-31 00:00:00', tz='UTC')
for check_dt in [boundary_dt, boundary_dt + pd.Timedelta(hours=1), boundary_dt - pd.Timedelta(hours=1)]:
    rn = df_native_trimmed[df_native_trimmed['date'] == check_dt]
    r1 = df_w1_trimmed[df_w1_trimmed['date'] == check_dt]
    r2 = df_w2_trimmed[df_w2_trimmed['date'] == check_dt]
    print(f'\n  At {check_dt}:')
    if len(rn): print(f'    Native: ema_f={rn.iloc[0]["ema_fast"]:.6f} ema_s={rn.iloc[0]["ema_slow"]:.6f} rsi={rn.iloc[0]["rsi"]:.4f}')
    else: print(f'    Native: MISSING')
    if len(r1): print(f'    W1:     ema_f={r1.iloc[0]["ema_fast"]:.6f} ema_s={r1.iloc[0]["ema_slow"]:.6f} rsi={r1.iloc[0]["rsi"]:.4f}')
    else: print(f'    W1:     MISSING')
    if len(r2): print(f'    W2:     ema_f={r2.iloc[0]["ema_fast"]:.6f} ema_s={r2.iloc[0]["ema_slow"]:.6f} rsi={r2.iloc[0]["rsi"]:.4f}')
    else: print(f'    W2:     MISSING')

# Compare signal counts
for label, df in [('Native', df_native_trimmed), ('W1', df_w1_trimmed), ('W2', df_w2_trimmed)]:
    el = (df.get('enter_long', 0) == 1).sum() if 'enter_long' in df.columns else 0
    es = (df.get('enter_short', 0) == 1).sum() if 'enter_short' in df.columns else 0
    xl = (df.get('exit_long', 0) == 1).sum() if 'exit_long' in df.columns else 0
    xs = (df.get('exit_short', 0) == 1).sum() if 'exit_short' in df.columns else 0
    print(f'\n{label}: enter_long={el} enter_short={es} exit_long={xl} exit_short={xs}')

bt.cleanup()
