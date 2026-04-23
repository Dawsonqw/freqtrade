# Rolling Backtest Toolkit for Freqtrade (Futures)

本目录用于实现“滚动分片回测 + 批量历史数据下载”，避免一次性全量加载导致 OOM。

## 设计目标

- 支持多币种（500+）多周期数据场景。
- 回测按时间窗口分片执行，每片后释放内存。
- 保留仓位与资金状态连续性（跨窗口不断开）。
- 不修改 `freqtrade/` 核心代码。

## 目录

- `config/`：配置模板
- `scripts/`：可执行脚本（下载、回测）
- `src/`：核心实现
- `docs/`：方案与说明
- `output/`：运行产物（默认不入库）

## 数据目录约定

- 主数据目录：`/data/freqtrade_data`
- 元数据目录：`/data/freqtrade_data/_meta`
  - `binance_futures_manifest.csv`
  - `binance_futures_coverage.csv`

## 安全说明

请通过环境变量注入 API 凭证，不要写入仓库文件：

- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`

## 快速开始

1. 先生成并小批量下载数据（10个币 smoke test）：

```bash
python user_custom/rolling_backtest/scripts/download_binance_futures.py \
  --data-dir /data/freqtrade_data \
  --meta-dir /data/freqtrade_data/_meta \
  --endpoint https://demo-fapi.binance.com \
  --quote USDT \
  --timeframes 5m 15m 1h 4h 1d 1w \
  --years 5 \
  --batch-size 10 \
  --max-pairs 10
```

2. 大批量后台下载（全量）：

```bash
bash user_custom/rolling_backtest/scripts/run_bulk_download_bg.sh \
  /data/freqtrade_data \
  /data/freqtrade_data/_meta \
  user_custom/rolling_backtest/output
```

3. 滚动回测（示例 14 天窗口）：

```bash
python user_custom/rolling_backtest/scripts/run_rolling_backtest.py \
  --config user_custom/rolling_backtest/config/rolling_backtest_config.example.json \
  --pairs-file /data/freqtrade_data/_meta/binance_futures_manifest.csv \
  --window-days 14 \
  --datadir /data/freqtrade_data \
  --user-data-dir /root/workspace/freqtrade/user_data \
  --output-json user_custom/rolling_backtest/output/rolling_backtest_result.json \
  --log-file user_custom/rolling_backtest/output/rolling_backtest.log
```
