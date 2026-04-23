# Parity Validation (Native vs Rolling)

## Validation command set

```bash
# rolling
python user_custom/rolling_backtest/scripts/run_rolling_backtest.py \
  --config user_custom/rolling_backtest/config/rolling_backtest_config.example.json \
  --strategy RollingSmokeStrategy \
  --timerange 20260101-20260401 \
  --window-days 14 \
  --datadir /data/freqtrade_data \
  --user-data-dir /root/workspace/freqtrade/user_data \
  --pairs-file /tmp/parity_pairs.txt \
  --output-json user_custom/rolling_backtest/output/parity_rolling_result.json \
  --log-file user_custom/rolling_backtest/output/parity_rolling.log \
  --max-failed-windows 2

# native
python -m freqtrade backtesting \
  --config user_custom/rolling_backtest/config/rolling_backtest_config.example.json \
  --strategy RollingSmokeStrategy \
  --timerange 20260101-20260401 \
  --pairs BTC/USDT:USDT \
  --datadir /data/freqtrade_data \
  --user-data-dir /root/workspace/freqtrade/user_data \
  --export trades

# compare
python user_custom/rolling_backtest/scripts/compare_parity.py \
  --rolling-json user_custom/rolling_backtest/output/parity_rolling_result.json \
  --native-json /root/workspace/freqtrade/user_data/backtest_results/backtest-result-2026-04-23_16-02-44.zip \
  --strategy RollingSmokeStrategy \
  --output-json user_custom/rolling_backtest/output/parity_report.json
```

## Latest result

- `all_passed: true`
- `total_trades: 173`
- `final_balance: 9125.6321723`
- `trade_digest: 026630e53725ee9b3fc35e9412bc735f5c25eb82d265c1136c65bd03a33fb66e`

## Notes

- Rolling runner now supports window-level fault tolerance (`--fail-fast` / `--max-failed-windows`).
- Downloader defaults to `--candle-types futures` to avoid unnecessary futures funding-rate pulls causing 403 in some endpoints.
- Full-night pipeline script: `scripts/run_nightly_pipeline.sh` (download + integrity check).
