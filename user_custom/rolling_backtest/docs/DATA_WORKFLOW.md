# 数据下载与增量更新说明

## 下载来源

- 交易所元数据（上市时间、合约列表）
  - `GET https://demo-fapi.binance.com/fapi/v1/exchangeInfo`
- K线历史
  - 由 freqtrade 下载器通过 Binance futures API 拉取并写入本地 datadir

## 本地落地位置

- OHLCV 数据：`/data/freqtrade_data`（freqtrade 原生格式）
- 元数据：`/data/freqtrade_data/_meta`
  - `binance_futures_manifest.csv`
  - `binance_futures_coverage.csv`

## manifest 字段

- `pair`：freqtrade pair（如 `BTC/USDT:USDT`）
- `symbol`：交易所符号（如 `BTCUSDT`）
- `onboard_date_utc`：Binance 上架时间
- `planned_start_utc`：下载起点（`max(上架时间, 当前-5年)`）

## 覆盖率 coverage 字段

- `pair,timeframe,candle_type,min_date,max_date,candles`

可用于后续增量下载策略：
- 对每个 `pair/timeframe`，下一次下载从 `max_date` 之后开始。
- 当前脚本通过 freqtrade 的更新逻辑自动续传。
