# Rolling Backtest Framework for Freqtrade (Futures)

滚动分片回测框架，支持 500+ 币种多年数据回测，自动内存管理。

## 核心能力

- **滚动窗口回测**：按时间窗口分片执行，自动根据系统内存计算最优窗口大小
- **跨窗口交易连续性**：开仓跨越窗口边界时自动延伸数据加载，保持指标连续
- **多策略对比**：单次运行多策略，生成对比报告（Sharpe/Sortino/胜率/回撤）
- **Walk-Forward Optimization (WFO)**：Optuna 驱动的滚动优化，含过拟合检测
- **并行指标计算**：ThreadPoolExecutor 加速 TA-Lib 指标（释放 GIL）
- **全量数据管道**：Binance USDT-M 合约数据下载 → 质量校验 → 可用性索引 → 回测
- **离线模式**：缓存交易所元数据，回测时无需网络连接
- **信号导出**：按窗口记录买卖信号，检测窗口边界信号偏差

## 目录结构

```
rolling_backtest/
├── src/                    # 核心引擎
│   ├── rolling_runner.py   # 主回测引擎 RollingBacktestRunner
│   ├── windowing.py        # 窗口规划 + 自动内存估算
│   ├── window_metrics.py   # 窗口级绩效指标 (Sharpe/Sortino/Calmar)
│   ├── comparison_report.py# 多策略对比报告
│   ├── signal_export.py    # 信号记录 + 边界偏差检测
│   ├── pair_filter.py      # 币对可用性过滤
│   ├── manifest.py         # 交易对元数据
│   ├── parity_tools.py     # 交易摘要 SHA-256 (一致性验证)
│   ├── wfo_engine.py       # WFO 编排器
│   ├── wfo_fold.py         # WFO 单折执行 (IS优化 → OOS验证)
│   ├── wfo_optimizer.py    # Optuna 超参数优化
│   ├── wfo_windowing.py    # WFO 折生成
│   └── wfo_report.py       # WFO 报告 + 过拟合分析
├── scripts/                # CLI 入口
│   ├── run_rolling_backtest.py    # 滚动回测主入口
│   ├── run_wfo.py                 # WFO 主入口
│   ├── download_binance_futures.py# 增量数据下载器
│   ├── check_data_quality.py      # 数据质量检查
│   ├── check_data_integrity.py    # 跨周期一致性校验
│   ├── build_pair_availability.py # 币对可用性索引
│   ├── cache_exchange_data.py     # 交易所离线缓存
│   ├── compare_parity.py          # 滚动 vs 原生一致性对比
│   └── generate_fake_data.py      # 合成测试数据
├── strategies/             # 策略
│   ├── FullMarketDynamicStrategy.py  # 全市场动态选币策略 (535+ pairs)
│   ├── ShortTermFiniteStrategy.py    # 短周期有限窗口策略
│   ├── FrameworkBenchStrategy.py     # 基准测试策略
│   ├── RollingSmokeStrategy.py       # 冒烟测试策略
│   └── WFOSmokeStrategy.py          # WFO 冒烟测试策略
├── config/                 # JSON 配置模板
├── tests/                  # 测试套件
├── docs/                   # 技术文档
│   ├── TECHNICAL_REFERENCE.md  # 完整 API + 架构参考 (agent 必读)
│   └── DATA_WORKFLOW.md        # 数据管道说明
└── output/                 # 运行产物 (不入库)
```

## 快速开始

### 1. 数据下载

```bash
# 生成 manifest (查看可下载的币对)
python scripts/download_binance_futures.py --manifest-only

# 全量下载 (后台)
bash scripts/run_bulk_download_bg.sh /data/freqtrade_data /data/freqtrade_data/_meta output

# 构建可用性索引
python scripts/build_pair_availability.py --data-dir /data/freqtrade_data --output /data/freqtrade_data/_meta/pair_availability.csv

# 缓存交易所元数据 (离线回测用)
python scripts/cache_exchange_data.py --output /data/freqtrade_data/_meta/exchange_cache.json
```

### 2. 滚动回测

```bash
python scripts/run_rolling_backtest.py \
  --config config/benchmark_config.json \
  --strategy FrameworkBenchStrategy \
  --timerange 20240101-20250101 \
  --window-days 0 \                    # 0=自动计算
  --datadir /data/freqtrade_data \
  --output-json output/result.json \
  --window-metrics \
  --parallel-workers 4
```

### 3. Walk-Forward 优化

```bash
python scripts/run_wfo.py \
  --config config/wfo_smoke_config.json \
  --strategy WFOSmokeStrategy \
  --is-days 90 --oos-days 30 \
  --n-epochs 100 \
  --loss-function SharpeHyperOptLoss
```

### 4. 一致性验证

```bash
python scripts/compare_parity.py \
  --rolling-json output/rolling_result.json \
  --native-json user_data/backtest_results/native.zip \
  --strategy FrameworkBenchStrategy
```

## 数据目录约定

- OHLCV 数据：`/data/freqtrade_data/futures/binance/*.feather`
- 元数据：`/data/freqtrade_data/_meta/`
  - `binance_futures_manifest.csv` — 币对清单
  - `binance_futures_coverage.csv` — 数据覆盖率
  - `pair_availability.csv` — 币对可用性索引
  - `exchange_cache.json` — 交易所离线缓存
  - `download_state.json` — 下载断点续传状态

## 安全说明

API 凭证通过环境变量注入，不写入仓库：`BINANCE_API_KEY` / `BINANCE_API_SECRET`

## 技术文档

详细的模块 API、架构设计、配置 Schema 见 [docs/TECHNICAL_REFERENCE.md](docs/TECHNICAL_REFERENCE.md)。
**Agent 在修改本框架代码前必须先读取该文档。**
