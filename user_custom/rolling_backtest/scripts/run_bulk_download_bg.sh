#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${1:-/data/freqtrade_data}
META_DIR=${2:-/data/freqtrade_data/_meta}
LOG_DIR=${3:-/root/workspace/freqtrade/user_custom/rolling_backtest/output}

mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/download_full_${TS}.log"

python /root/workspace/freqtrade/user_custom/rolling_backtest/scripts/download_binance_futures.py \
  --data-dir "$DATA_DIR" \
  --meta-dir "$META_DIR" \
  --endpoint "https://demo-fapi.binance.com" \
  --quote USDT \
  --years 5 \
  --timeframes 5m 15m 1h 4h 1d 1w \
  --batch-size 30 \
  > "$LOG_FILE" 2>&1

echo "done log=$LOG_FILE"
