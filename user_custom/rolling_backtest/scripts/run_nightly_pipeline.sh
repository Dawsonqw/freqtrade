#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${1:-/data/freqtrade_data}
META_DIR=${2:-/data/freqtrade_data/_meta}
OUT_DIR=${3:-/root/workspace/freqtrade/user_custom/rolling_backtest/output}

mkdir -p "$OUT_DIR" "$META_DIR"
TS=$(date +%Y%m%d_%H%M%S)
PIPE_LOG="$OUT_DIR/nightly_pipeline_${TS}.log"

{
  echo "[nightly] start ts=$TS"

  python /root/workspace/freqtrade/user_custom/rolling_backtest/scripts/download_binance_futures.py \
    --data-dir "$DATA_DIR" \
    --meta-dir "$META_DIR" \
    --endpoint "https://fapi.binance.com" \
    --quote USDT \
    --years 5 \
    --timeframes 5m 15m 1h 4h 1d 1w \
    --batch-size 30 \
    --candle-types futures \
    --prepend \
    --retries 4 \
    --retry-sleep 2 \
    --sleep-between-batches 0.2 \
    --strict-failures

  python /root/workspace/freqtrade/user_custom/rolling_backtest/scripts/check_data_integrity.py \
    --data-dir "$DATA_DIR" \
    --meta-dir "$META_DIR" \
    --timeframes 5m 15m 1h 4h 1d 1w \
    --min-coverage 0.95 \
    --grace-candles 200

  echo "[nightly] done ts=$(date +%Y%m%d_%H%M%S)"
} 2>&1 | tee "$PIPE_LOG"

echo "pipeline_log=$PIPE_LOG"
