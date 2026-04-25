#!/usr/bin/env python3
import json, zipfile, pandas as pd, sys
from collections import Counter

def load_trades(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith('.json') and not name.endswith('_config.json') and 'meta' not in name:
                data = json.loads(zf.read(name))
                if 'strategy' in data:
                    for sname, sdata in data['strategy'].items():
                        return pd.DataFrame(sdata['trades'])
    return pd.DataFrame()

native = load_trades('user_data/backtest_results/backtest-result-2026-04-25_10-25-09.zip')
rolling = load_trades('user_data/backtest_results/backtest-result-2026-04-25_10-28-20.zip')

print(f'Native: {len(native)} trades')
print(f'Rolling: {len(rolling)} trades')

for df in [native, rolling]:
    df['open_date'] = pd.to_datetime(df['open_date'], utc=True)
    df['close_date'] = pd.to_datetime(df['close_date'], utc=True)
    df.sort_values('open_date', inplace=True)
    df.reset_index(drop=True, inplace=True)

def make_key(row):
    return f"{row['pair']}_{row['open_timestamp']}"

native['key'] = native.apply(make_key, axis=1)
rolling['key'] = rolling.apply(make_key, axis=1)

native_keys = set(native['key'])
rolling_keys = set(rolling['key'])

only_native = native_keys - rolling_keys
only_rolling = rolling_keys - native_keys
both = native_keys & rolling_keys

print(f'\nMatched (same pair+open_ts): {len(both)}')
print(f'Only in native: {len(only_native)}')
print(f'Only in rolling: {len(only_rolling)}')

boundary = pd.Timestamp('2024-01-31', tz='UTC')

if only_rolling:
    r_only = rolling[rolling['key'].isin(only_rolling)].copy()
    print(f'\n=== Only-in-rolling trades: {len(r_only)} ===')
    before = r_only[r_only['open_date'] < boundary]
    after = r_only[r_only['open_date'] >= boundary]
    print(f'  Before boundary (Jan 31): {len(before)}')
    print(f'  After boundary (>=Jan 31): {len(after)}')
    around = r_only[(r_only['open_date'] >= '2024-01-25') & (r_only['open_date'] <= '2024-02-05')]
    print(f'  Around boundary (Jan 25 - Feb 5): {len(around)}')
    print('\n  First 10 only-in-rolling:')
    for _, row in r_only.head(10).iterrows():
        print(f'    {row["pair"]} open={row["open_date"]} close={row["close_date"]} exit={row["exit_reason"]} pnl={row["profit_ratio"]:.6f}')

if only_native:
    n_only = native[native['key'].isin(only_native)].copy()
    print(f'\n=== Only-in-native trades: {len(n_only)} ===')
    print('  First 10:')
    for _, row in n_only.head(10).iterrows():
        print(f'    {row["pair"]} open={row["open_date"]} close={row["close_date"]} exit={row["exit_reason"]} pnl={row["profit_ratio"]:.6f}')

if both:
    n_matched = native[native['key'].isin(both)].copy().set_index('key')
    r_matched = rolling[rolling['key'].isin(both)].copy().set_index('key')
    
    diff_count = 0
    diff_types = Counter()
    for key in sorted(both):
        n = n_matched.loc[key]
        r = r_matched.loc[key]
        if isinstance(n, pd.DataFrame): n = n.iloc[0]
        if isinstance(r, pd.DataFrame): r = r.iloc[0]
        diffs = []
        if abs(n['close_timestamp'] - r['close_timestamp']) > 60000:
            diffs.append('close_time')
        if n['exit_reason'] != r['exit_reason']:
            diffs.append('exit_reason')
        if abs(n['profit_ratio'] - r['profit_ratio']) > 0.0001:
            diffs.append('pnl')
        if diffs:
            diff_count += 1
            diff_types.update(diffs)
            if diff_count <= 5:
                print(f'\n  Diff #{diff_count}: {key}')
                print(f'    Native:  close={n["close_date"]} exit={n["exit_reason"]} pnl={n["profit_ratio"]:.6f}')
                print(f'    Rolling: close={r["close_date"]} exit={r["exit_reason"]} pnl={r["profit_ratio"]:.6f}')
    
    print(f'\n=== Matched trades with differences: {diff_count}/{len(both)} ===')
    print(f'  Diff types: {dict(diff_types)}')
