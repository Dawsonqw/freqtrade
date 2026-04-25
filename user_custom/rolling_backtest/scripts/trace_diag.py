#!/usr/bin/env python3
"""Trace exactly how native vs rolling produce different trades from same signals."""
import sys, json
sys.path.insert(0, '/root/workspace/freqtrade')

import pandas as pd
from freqtrade.configuration import Configuration, TimeRange
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.data import history
from freqtrade.data.converter import trim_dataframes
from freqtrade.persistence import LocalTrade

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
strat = bt.strategylist[0]
bt._set_strategy(strat)

# Load data (same as native backtest)
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
print(f'Raw data: {len(data[pair])} candles')

# Compute indicators (same as native)
preprocessed = strat.advise_all_indicators(data)
preprocessed_tmp = trim_dataframes(preprocessed, full_tr, bt.required_startup)

if not preprocessed_tmp:
    print("ERROR: empty after trim")
    sys.exit(1)

min_date, max_date = history.get_timerange(preprocessed_tmp)
print(f'After trim: min_date={min_date}, max_date={max_date}')
print(f'preprocessed[pair] rows: {len(preprocessed[pair])}')
print(f'preprocessed_tmp[pair] rows: {len(preprocessed_tmp[pair])}')

# Convert to lists (same as native)
processed_lists = bt._get_ohlcv_as_lists(preprocessed)
bt._load_bt_data_detail()

# Now do the backtest loop EXACTLY as native does
bt.reset_backtest(bt.enable_protections)
bt.wallets.update()

print(f'\n=== Running NATIVE-style backtest loop ===')
for current_time, pair_name, row, is_last_row, trade_dir in bt.time_pair_generator(
    min_date, max_date, list(processed_lists.keys()), processed_lists
):
    if not bt._can_short or trade_dir is None:
        bt.backtest_loop(row, pair_name, current_time, trade_dir, not is_last_row)
    else:
        for _ in (0, 1):
            closed_dir = bt.backtest_loop(
                row, pair_name, current_time, trade_dir, not is_last_row
            )
            if not closed_dir or closed_dir == trade_dir:
                break

bt.handle_left_open(LocalTrade.bt_trades_open_pp, data=processed_lists)

native_trades = len(LocalTrade.bt_trades)
native_open = len(LocalTrade.bt_trades_open)
print(f'Native-style: {native_trades} trades, {native_open} open')

# Show first 5 trades
for i, t in enumerate(LocalTrade.bt_trades[:5]):
    print(f'  {i}: {t.pair} open={t.open_date} close={t.close_date} exit={t.exit_reason} pnl={t.close_profit:.6f}')

# Now simulate rolling: 2 windows
print(f'\n=== Running ROLLING-style backtest loop (2 windows) ===')
bt._set_strategy(strat)
bt.reset_backtest(bt.enable_protections)
bt.wallets.update()

from freqtrade.exchange import timeframe_to_seconds
startup_secs = bt.required_startup * timeframe_to_seconds('5m')

windows = [
    ('W0', '2024-01-01', '2024-01-31'),
    ('W1', '2024-01-31', '2024-03-01'),
]

for widx, (wname, wstart_str, wend_str) in enumerate(windows):
    wstart = pd.Timestamp(wstart_str, tz='UTC')
    wend = pd.Timestamp(wend_str, tz='UTC')
    
    wtr = TimeRange('date', 'date', int(wstart.timestamp()), int(wend.timestamp()))
    bt.timerange = wtr
    
    # Slice data with startup extension (same as _slice_preloaded_data)
    data_start = wstart - pd.Timedelta(seconds=startup_secs)
    mask = (data[pair]['date'] >= data_start) & (data[pair]['date'] <= wend)
    window_data = {pair: data[pair][mask].copy()}
    
    # Compute indicators
    w_preprocessed = strat.advise_all_indicators(window_data)
    w_trimmed = trim_dataframes(w_preprocessed, wtr, bt.required_startup)
    
    if not w_trimmed:
        print(f'  {wname}: empty after trim')
        continue
    
    w_min, w_max = history.get_timerange(w_trimmed)
    w_lists = bt._get_ohlcv_as_lists(w_preprocessed)
    bt._load_bt_data_detail()
    
    is_last = (widx == len(windows) - 1)
    
    print(f'  {wname}: data={len(window_data[pair])} preprocessed={len(w_preprocessed[pair])} trimmed={len(w_trimmed[pair])} min={w_min} max={w_max}')
    
    for current_time, pair_name, row, is_last_row, trade_dir in bt.time_pair_generator(
        w_min, w_max, list(w_lists.keys()), w_lists
    ):
        if not bt._can_short or trade_dir is None:
            bt.backtest_loop(row, pair_name, current_time, trade_dir, not is_last_row)
        else:
            for _ in (0, 1):
                closed_dir = bt.backtest_loop(
                    row, pair_name, current_time, trade_dir, not is_last_row
                )
                if not closed_dir or closed_dir == trade_dir:
                    break
    
    if is_last:
        bt.handle_left_open(LocalTrade.bt_trades_open_pp, data=w_lists)
    
    rolling_trades = len(LocalTrade.bt_trades)
    rolling_open = len(LocalTrade.bt_trades_open)
    print(f'  {wname} done: total_trades={rolling_trades}, open={rolling_open}')

print(f'\nRolling-style: {len(LocalTrade.bt_trades)} trades total')

# Show first 5 trades
for i, t in enumerate(LocalTrade.bt_trades[:5]):
    print(f'  {i}: {t.pair} open={t.open_date} close={t.close_date} exit={t.exit_reason} pnl={t.close_profit:.6f}')

bt.cleanup()
