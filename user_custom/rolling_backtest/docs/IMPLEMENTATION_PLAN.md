# Rolling Futures Backtest Framework (Chunked) — Implementation Plan

> For Hermes: keep all code under `user_custom/rolling_backtest/` and do not modify core freqtrade modules.

## Goal

在 8G 服务器上实现可运行的“滚动多币种回测框架”：分片加载数据、分片计算指标、分片执行回测，同时保持仓位和资金状态连续，避免一次性加载 500+ 合约多周期数据导致 OOM。

## Architecture

- 复用 `freqtrade.optimize.backtesting.Backtesting` 的核心撮合/风控/统计逻辑。
- 使用外层“窗口调度器”按时间片加载数据（带 startup overlap），每片完成后释放 DataFrame 内存。
- 仅在最后窗口完成时统一平仓与生成报告，保证连续权益曲线和仓位。

## Scope

1. 新建独立目录与文档（不污染主项目）。
2. 新增 Binance Futures 数据下载脚本（5m/15m/1h/4h/1d/1w，最近 5 年，上市不足 5 年按上市日起）。
3. 新增滚动回测执行器（chunk run，连续状态）。
4. 提供小批量验证与后台大批量执行命令。

## Directory Layout

- `user_custom/rolling_backtest/README.md`
- `user_custom/rolling_backtest/config/rolling_backtest_config.example.json`
- `user_custom/rolling_backtest/scripts/download_binance_futures.py`
- `user_custom/rolling_backtest/scripts/run_rolling_backtest.py`
- `user_custom/rolling_backtest/src/rolling_runner.py`
- `user_custom/rolling_backtest/src/windowing.py`
- `user_custom/rolling_backtest/src/manifest.py`
- `user_custom/rolling_backtest/output/` (运行产物，gitignore)

## Milestones

1. Scaffold + README + config example.
2. 下载脚本（先 `--limit-pairs 10` 小批量验证）。
3. 滚动回测引擎（先单策略）。
4. 端到端 smoke test。
5. 文档补齐 + Git 提交。

## Validation

- 下载阶段：
  - 生成 `manifest.csv`（symbol、onboard_date、planned_start、local_path）。
  - 生成 `coverage.csv`（pair/timeframe/min_dt/max_dt/candles）。
- 回测阶段：
  - 每窗口日志包含：窗口范围、加载 pair 数、内存峰值估计、耗时。
  - 最终输出包含：总交易数、最终权益、窗口执行统计。

## Non-goals (for this iteration)

- 不改 freqtrade 内核。
- 不在本轮实现分布式并行回测。
- 不在仓库提交真实 API key/secret（仅保留模板与环境变量说明）。
