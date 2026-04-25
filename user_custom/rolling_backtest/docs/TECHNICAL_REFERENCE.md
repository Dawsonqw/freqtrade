# Rolling Backtest Framework — Technical Reference

> Auto-generated comprehensive API and architecture reference.

---

## 1. Architecture Overview

### Data Flow Diagram

```
CLI Entry Points
┌──────────────────────────────────────────────────────────────────┐
│ scripts/run_rolling_backtest.py  →  RollingBacktestRunner.run()  │
│ scripts/run_wfo.py              →  WFOEngine.run()               │
└─────────┬────────────────────────────────┬───────────────────────┘
          │                                │
          ▼                                ▼
┌─────────────────────┐     ┌──────────────────────────────┐
│  Rolling Backtest    │     │  Walk-Forward Optimization    │
│  Pipeline            │     │  Pipeline                     │
│                      │     │                               │
│  windowing.py        │     │  wfo_windowing.py             │
│  ├→ auto_window_days │     │  ├→ build_wfo_folds           │
│  └→ build_windows    │     │  │                            │
│                      │     │  wfo_optimizer.py              │
│  rolling_runner.py   │     │  ├→ Optuna IS optimization    │
│  ├→ _preload_all_data│     │  │                            │
│  ├→ _slice_preloaded │     │  wfo_fold.py                  │
│  ├→ _run_single_win  │     │  ├→ IS optimize → OOS validate│
│  └→ _run_strategy    │     │  │                            │
│                      │     │  wfo_report.py                 │
│  window_metrics.py   │     │  └→ overfit analysis           │
│  comparison_report.py│     └──────────────────────────────┘
│  signal_export.py    │
│  pair_filter.py      │
│  parity_tools.py     │
│  manifest.py         │
└─────────────────────┘

Data Acquisition Scripts
┌──────────────────────────────────────────────────────────────────┐
│ download_binance_futures.py  →  Binance FAPI → feather files     │
│ check_data_quality.py        →  gap/staleness detection          │
│ check_data_integrity.py      →  cross-timeframe consistency      │
│ build_pair_availability.py   →  pair_availability.csv            │
│ cache_exchange_data.py       →  exchange_cache.json (offline)    │
│ generate_fake_data.py        →  synthetic OHLCV for testing      │
└──────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Chunked Windowed Execution**: Splits long backtests into memory-safe time windows, preserving trade state across boundaries via `LocalTrade.bt_trades_open`.
2. **Data Pre-loading**: Loads full OHLCV once, then slices per window via pandas indexing — avoids repeated disk I/O.
3. **Auto Window Sizing**: Uses memory estimation (pairs × candles × indicator multiplier) and system RAM to automatically pick optimal window size.
4. **Trade State Continuity**: Open trades carry over between windows. Extended timerange loads cover open trades' entry points for continuous indicator calculation.
5. **Aggressive Memory Management**: `gc.collect()` after every window, explicit `del` of DataFrames, cache clearing.
6. **Offline Exchange Init**: Monkey-patches `Exchange.reload_markets()` to inject cached market data, skipping network calls.
7. **Parallel Indicators**: ThreadPoolExecutor for TA-Lib (which releases GIL), with serial post-processing for cross-pair ranking.

---

## 2. Module-by-Module API Reference

### 2.1 `src/rolling_runner.py` — Main Engine

#### `WindowRunStat` (dataclass)
```python
@dataclass
class WindowRunStat:
    index: int
    start: str              # ISO datetime
    end: str
    pairs_loaded: int
    candles_total: int
    trades_after_window: int
    open_trades_after_window: int
    duration_sec: float
    status: str             # "ok" | "no_data" | "error"
    error: str | None = None
    carry_over_trades: int = 0
```

#### `RollingBacktestRunner`
```python
class RollingBacktestRunner:
    def __init__(
        self,
        backtesting: Backtesting,
        window_days: int = 0,           # 0 = auto-calculate
        *,
        mem_budget_mb: float = 0,       # 0 = auto-detect 50% free RAM
        preferred_days: int = 0,        # hint for auto mode
        min_window_days: int = 7,
        max_window_days: int = 90,
        parallel_workers: int = 0,      # 0 = serial
    ) -> None

    # Data loading
    def _preload_all_data(self, full_tr: TimeRange) -> None
    def _slice_preloaded_data(self, tr: TimeRange, window: Window | None = None) -> dict[str, DataFrame]
    def _load_window_data(self, tr: TimeRange, window: Window | None = None) -> dict[str, DataFrame]

    # Parallelism
    def _parallel_advise_all_indicators(self, data: dict[str, DataFrame], workers: int) -> dict[str, DataFrame]
    def _compute_rankings(self, strategy, result: dict[str, DataFrame]) -> None

    # Execution
    def _get_extended_timerange_for_open_trades(self, window: Window) -> TimeRange
    def _refresh_pairlist_for_window(self, window: Window) -> list[str]
    def _run_single_window(self, *, window: Window, is_last_window: bool, signal_exporter: SignalExporter | None = None) -> None

    # Main entry
    def run(self, output_json: Path | None = None, *, fail_fast: bool = False, max_failed_windows: int = 0, export: str = "none", plot: bool = False, export_signals: bool = False) -> dict

    def _run_strategy(self, *, strat, fail_fast: bool, max_failed_windows: int, export_signals: bool = False) -> tuple[dict, dict | None]

    # Reporting
    def _print_window_summary_table(self, strategy_name: str, stake_currency: str, window_performances: list[WindowPerformance] | None = None) -> None
    def _generate_standard_reports(self, *, all_bt_content: dict, min_date: datetime, max_date: datetime, started_at: datetime, ended_at: datetime, export: str, plot: bool = False) -> None
    def _generate_plots(self, btdata: dict[str, DataFrame], all_bt_content: dict, min_date: datetime, max_date: datetime) -> None
```

**Key Implementation Details:**
- Pre-loads data once when `len(windows) > 1`, caches tick_size per pair lazily
- Window execution: load data → compute indicators → trim to window → `backtest_loop()` per candle → handle_left_open on last window
- Multi-strategy: iterates `bt.strategylist`, builds comparison report if >1 strategy
- Output: JSON result + freqtrade standard ZIP + optional HTML profit plots

---

### 2.2 `src/windowing.py` — Window Planning

#### `Window` (frozen dataclass)
```python
@dataclass(frozen=True)
class Window:
    index: int
    start: datetime
    end: datetime
    timerange: TimeRange
```

#### `WindowPlan` (frozen dataclass)
```python
@dataclass(frozen=True)
class WindowPlan:
    window_days: int
    num_windows: int
    total_days: int
    num_pairs: int
    mem_per_window_mb: float
    mem_budget_mb: float
    max_window_days: int
    reason: str
```

#### Functions
```python
def resolve_global_bounds(timerange: TimeRange, timeframe: str) -> tuple[datetime, datetime]
def estimate_window_memory_mb(num_pairs: int, window_days: int, timeframe: str = "5m") -> float
def auto_window_days(timerange: TimeRange, *, timeframe: str = "5m", num_pairs: int, mem_budget_mb: float = 0, preferred_days: int = 0, min_window_days: int = 7, max_window_days: int = 90) -> WindowPlan
def build_windows(timerange: TimeRange, *, timeframe: str, window_days: int = 0, num_pairs: int = 0, mem_budget_mb: float = 0, preferred_days: int = 0, min_window_days: int = 7, max_window_days: int = 90) -> tuple[list[Window], Optional[WindowPlan]]
```

**Constants:**
- `_CANDLES_PER_DAY_5M = 288`
- `_BYTES_PER_CANDLE = 60`
- `_INDICATOR_MULTIPLIER = 2.5`

**Auto-sizing Algorithm:**
1. Compute total_days from timerange
2. Get memory budget (50% free RAM via psutil, or 2GB fallback)
3. Calculate max affordable days = budget / (mem_per_day × pairs)
4. If preferred_days given + fits: use it or find nearby divisor (±5)
5. Otherwise: find largest exact divisor of total_days in [min, max]
6. Fallback: minimize tail window waste (tail ≥ 50% of window)
7. Merges degenerate tail windows (<1 day) into previous window

---

### 2.3 `src/window_metrics.py` — Performance Metrics

#### `WindowPerformance` (dataclass)
```python
@dataclass
class WindowPerformance:
    window_index: int
    start: str
    end: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_profit_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float        # as fraction (0.0 - 1.0)
    profit_factor: float
    carry_over_trades: int = 0

    def to_dict(self) -> dict
```

#### Functions
```python
def compute_window_performance(trades: DataFrame, window_index: int, start: datetime, end: datetime, starting_balance: float, carry_over_trades: int = 0) -> WindowPerformance
def compute_all_window_performances(all_trades: DataFrame, window_boundaries: list[tuple[int, datetime, datetime]], starting_balance: float) -> list[WindowPerformance]
```

**Metrics computed:** Sharpe ratio (annualized, 365 periods), Sortino ratio, max drawdown (cumulative P&L based), profit factor, win rate.

---

### 2.4 `src/comparison_report.py` — Multi-Strategy Comparison

#### `StrategyComparison` (dataclass)
```python
@dataclass
class StrategyComparison:
    strategy_name: str
    total_trades: int
    total_profit_pct: float
    avg_sharpe: float
    avg_sortino: float
    avg_win_rate: float
    worst_drawdown: float
    avg_profit_factor: float
    consistency_score: float    # 1 - std(sharpes)/mean(sharpes), higher=better
    windows_with_profit: int
    windows_total: int
```

#### `ComparisonReport` (dataclass)
```python
@dataclass
class ComparisonReport:
    strategies: list[StrategyComparison]
    recommended: str            # best strategy name
    recommendation_reason: str

    def to_dict(self) -> dict
```

#### Functions
```python
def build_comparison_report(strategy_performances: dict[str, list[WindowPerformance]]) -> ComparisonReport
def format_comparison_table(report: ComparisonReport) -> str
```

**Ranking:** Composite score = 0.3×avg_sharpe + 0.2×avg_win_rate×100 + 0.2×consistency×10 + 0.15×avg_profit_factor + 0.15×(1-worst_drawdown)×10

---

### 2.5 `src/signal_export.py` — Signal Recording

#### `SignalExporter`
```python
class SignalExporter:
    SIGNAL_COLUMNS = ["date", "enter_long", "enter_short", "exit_long", "exit_short", "enter_tag", "exit_tag"]

    def __init__(self, output_dir: Path) -> None
    def collect_window(self, window_index: int, preprocessed: dict[str, DataFrame]) -> None
    def export(self) -> Path                # returns CSV path
    def export_deviations(self) -> Path | None  # returns CSV path or None
```

**Deviation Detection:** Compares signals at window boundaries (last 5 rows of window N vs first 5 rows of window N+1 for same pair+date). Reports mismatches as CSV.

---

### 2.6 `src/pair_filter.py` — Pair Availability Filtering

#### `PairAvailabilityFilter`
```python
class PairAvailabilityFilter:
    DEFAULT_PATH = Path("/data/freqtrade_data/_meta/pair_availability.csv")

    def __init__(self, availability_csv: Path | None = None) -> None

    @property
    def loaded(self) -> bool

    def filter_pairs(self, pairs: list[str], *, window_start: str, window_end: str, timeframe: str) -> list[str]
```

**CSV Format:** `pair,timeframe,first_date,last_date,candle_count`
Filters pairs to those with data coverage overlapping the window's time range.

---

### 2.7 `src/manifest.py` — Symbol Metadata

#### `SymbolMeta` (frozen dataclass)
```python
@dataclass(frozen=True)
class SymbolMeta:
    pair: str              # e.g. "BTC/USDT:USDT"
    symbol: str            # e.g. "BTCUSDT"
    onboard_date: datetime
    planned_start: datetime
    quote: str             # e.g. "USDT"
```

#### Functions
```python
def symbol_to_pair(symbol: str, quote: str) -> str
def compute_planned_start(onboard_ms: int, years: int) -> datetime
def build_symbol_meta(symbols: Iterable[dict], quote: str, years: int) -> list[SymbolMeta]
def write_manifest(rows: list[SymbolMeta], path: Path) -> None
def read_manifest(path: Path) -> list[SymbolMeta]
```

Filters: contractType=PERPETUAL, status=TRADING, quoteAsset matches. planned_start = max(onboard_date, now - years).

---

### 2.8 `src/parity_tools.py` — Trade Digest

#### Constants
```python
DEFAULT_TRADE_COLUMNS = ["pair", "open_date", "close_date", "open_rate", "close_rate", "profit_ratio", "is_short", "exit_reason"]
```

#### Functions
```python
def compute_trade_digest(trades: DataFrame, columns: list[str] | None = None) -> str  # SHA-256 hex
def load_backtest_json(path: str | Path) -> dict | list  # supports .zip
def load_trades_from_backtest_json(path: str | Path, strategy: str | None = None) -> DataFrame
```

Canonical normalization: datetimes → ISO format, floats → 10 decimal places, sorted deterministically. Used for bitwise-reproducibility verification.

---

### 2.9 WFO Subsystem

#### `src/wfo_windowing.py`

```python
@dataclass(frozen=True)
class WFOFold:
    index: int
    is_start: datetime
    is_end: datetime
    oos_start: datetime
    oos_end: datetime

    @property
    def is_timerange(self) -> TimeRange
    @property
    def oos_timerange(self) -> TimeRange
    @property
    def is_days(self) -> float
    @property
    def oos_days(self) -> float

def build_wfo_folds(timerange: TimeRange, *, timeframe: str, is_days: int, oos_days: int, step_days: int | None = None) -> list[WFOFold]
```

Sliding window: IS window of `is_days`, immediately followed by OOS of `oos_days`. Step defaults to `oos_days` (non-overlapping OOS).

#### `src/wfo_optimizer.py`

```python
@dataclass
class OptimizationResult:
    fold_index: int
    best_params: dict[str, Any]
    best_loss: float
    n_trials: int
    is_stats: dict[str, Any]
    params_details: dict[str, Any] = field(default_factory=dict)

class WFOOptimizer:
    def __init__(self, backtesting: Backtesting, *, n_epochs: int = 100, loss_function: str = "SharpeHyperOptLoss", spaces: list[str] | None = None) -> None
    def _load_and_prepare_data(self, timerange: TimeRange) -> tuple[dict[str, DataFrame], datetime, datetime]
    def _apply_params(self, params: dict[str, Any]) -> None
    def _get_dimensions(self) -> dict[str, Any]  # {"name": {"type": "int"|"float"|"categorical", ...}}
    def _suggest_params(self, trial: optuna.Trial, dimensions: dict) -> dict[str, Any]
    def optimize_fold(self, fold: WFOFold) -> OptimizationResult
```

Uses Optuna `create_study(direction="minimize")`. Loads IS data once per fold, runs `n_epochs` trials. Supports futures mode (funding rates + mark prices).

#### `src/wfo_fold.py`

```python
@dataclass
class FoldResult:
    fold_index: int
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str
    best_params: dict[str, Any]
    best_loss: float
    n_trials: int
    is_metrics: dict[str, Any]
    oos_metrics: dict[str, Any]
    efficiency_ratio: float | None = None  # OOS_profit / IS_profit
    status: str = "ok"
    error: str | None = None

class WFOFoldRunner:
    def __init__(self, backtesting: Backtesting, optimizer: WFOOptimizer) -> None
    def _run_oos_backtest(self, fold: WFOFold, params: dict[str, Any]) -> dict[str, Any]
    def run_fold(self, fold: WFOFold) -> FoldResult
```

Metrics extracted: profit_total, profit_total_abs, trade_count, win_rate, sharpe, sortino, max_drawdown, max_drawdown_abs, profit_factor, avg_duration.

#### `src/wfo_engine.py`

```python
class WFOEngine:
    def __init__(self, config: dict[str, Any], *, is_days: int = 90, oos_days: int = 30, step_days: int | None = None, n_epochs: int = 100, loss_function: str = "SharpeHyperOptLoss", spaces: list[str] | None = None, output_dir: str | Path | None = None) -> None
    def run(self, timerange_str: str | None = None) -> dict[str, Any]
```

Pipeline: build folds → init Backtesting + WFOOptimizer (once) → for each fold: IS optimize → OOS validate → gc.collect() → generate report.

#### `src/wfo_report.py`

```python
@dataclass
class WFOSummary:
    total_folds: int
    ok_folds: int
    failed_folds: int
    is_total_profit: float
    oos_total_profit: float
    oos_total_trades: int
    avg_efficiency_ratio: float | None
    overfit_score: float       # 0=perfect, 1=total overfit
    param_stability: float     # 0=unstable, 1=stable

def generate_wfo_report(folds: list[FoldResult], output_path: Path | None = None) -> dict[str, Any]
```

**Overfit Score:** `1 - (OOS_profit / IS_profit)`, clamped [0, 1].
**Param Stability:** Mean of `max(0, 1 - CV)` across all numeric parameter keys, where CV = coefficient of variation.
**Verdicts:** LOW_OVERFIT (<0.3), MODERATE_OVERFIT (0.3-0.6), HIGH_OVERFIT (≥0.6).

---

## 3. Scripts Reference

### 3.1 `scripts/run_rolling_backtest.py` — Main CLI

```
python run_rolling_backtest.py \
  --config CONFIG_PATH \
  [--strategy NAME] \
  [--strategy-list S1 S2 ...] \
  [--timerange 20230101-20240101] \
  [--window-days 30]              # 0=auto
  [--mem-budget-mb 4096]          # 0=auto (50% free RAM)
  [--preferred-days 30]           # hint for auto
  [--min-window-days 7] [--max-window-days 90]
  [--datadir /data/freqtrade_data]
  [--user-data-dir /root/workspace/freqtrade/user_data]
  [--pairs-file pairs.csv]
  [--output-json output/result.json]
  [--log-file run.log]
  [--fail-fast] [--max-failed-windows 3]
  [--export none|trades|signals]
  [--enable-protections]
  [--timeframe-detail 1m]
  [--backtest-breakdown day week month]
  [--plot]
  [--export-signals]
  [--parallel-workers 4]          # 0=serial
  [--window-metrics]
```

**Features:** Offline exchange init via cached markets, CSV/TXT pair loading, multi-strategy comparison, freqtrade standard ZIP export, HTML profit plots.

### 3.2 `scripts/run_wfo.py` — WFO CLI

```
python run_wfo.py \
  --config CONFIG_PATH \
  [--strategy NAME] \
  [--is-days 90] [--oos-days 30] [--step-days 30] \
  [--n-epochs 100] \
  [--loss-function SharpeHyperOptLoss] \
  [--spaces buy sell] \
  [--timerange 20230101-20240101] \
  [--datadir /data/freqtrade_data] \
  [--pairs-file pairs.csv] \
  [--output-dir wfo_results/] \
  [--log-file wfo.log]
```

### 3.3 `scripts/download_binance_futures.py` — Data Downloader

```
python download_binance_futures.py \
  [--data-dir /data/freqtrade_data] \
  [--meta-dir /data/freqtrade_data/_meta] \
  [--endpoint https://fapi.binance.com] \
  [--quote USDT] [--years 5] \
  [--timeframes 5m 15m 1h 4h 1d 1w] \
  [--batch-size 40] [--max-pairs 3] \
  [--manifest-only] [--coverage-only] \
  [--reset-state] [--prepend] [--retry-failed] \
  [--retries 5] [--retry-sleep 3.0] \
  [--sleep-between-batches 0] [--save-interval 1]
```

**Features:**
- `DownloadState` class: persistent JSON state file tracking `(pair, timeframe)` completion
- Graceful SIGINT/SIGTERM shutdown
- Exponential backoff retries
- Atomic state file writes
- Grouped by timeframe for efficient batching
- Generates manifest CSV + coverage CSV + download_report.json

### 3.4 `scripts/check_data_quality.py`

Scans feather files for gaps, staleness, zero-volume candles. Outputs quality report CSV.

### 3.5 `scripts/check_data_integrity.py`

Cross-timeframe consistency checks (e.g., 5m candle count should be 3× 15m count). Reports integrity violations.

### 3.6 `scripts/build_pair_availability.py`

Builds `pair_availability.csv` from feather files: `pair,timeframe,first_date,last_date,candle_count`. Used by `PairAvailabilityFilter`.

### 3.7 `scripts/cache_exchange_data.py`

Saves `exchange_cache.json` (all markets data) for offline Backtesting init without network access.

### 3.8 `scripts/compare_parity.py`

Compares trade digests between two backtest result files to verify reproducibility.

### 3.9 `scripts/generate_fake_data.py`

Generates synthetic OHLCV feather files for testing. Random walk price model.

### 3.10 Shell Scripts

- **`run_bulk_download_bg.sh`**: Background wrapper for download_binance_futures.py with nohup
- **`run_download.sh`**: Full download pipeline (manifest → download → coverage → quality check)
- **`run_nightly_pipeline.sh`**: Automated nightly: download → quality check → integrity check → backtest

---

## 4. Strategies

### 4.1 `FullMarketDynamicStrategy(IStrategy)`
- **Timeframe:** 15m, can_short=True, startup_candle_count=40
- **Core:** Full-market dynamic pair selection (535+ pairs). Pre-computes per-candle market rankings in `advise_all_indicators()` (vectorized). `confirm_trade_entry()` checks O(1) cache lookup.
- **Ranking:** Weighted composite: volatility_pct(40%) + volume_usd(30%) + ADX(30%), top 20 pairs
- **Entry:** EMA(9) crosses EMA(21) + RSI(14) in range + volume/volatility/ADX thresholds
- **Exit:** EMA reversal or RSI extreme + trailing stop
- **Risk:** stoploss=-3%, trailing_stop_positive=1%, max_open_trades=20, stake_amount=200 USDT

### 4.2 `ShortTermFiniteStrategy(IStrategy)`
- **Timeframe:** 5m, can_short=True, startup_candle_count=50
- **Core:** Short-term mean reversion + momentum with Bollinger Bands, RSI, MACD
- **Entry:** BB touch + RSI divergence + volume confirmation
- **Exit:** BB mean reversion + RSI normalization + trailing
- **Risk:** stoploss=-2%, trailing_stop_positive=0.8%

### 4.3 `FrameworkBenchStrategy(IStrategy)`
- **Timeframe:** 5m, can_short=True, startup_candle_count=50
- **Core:** Benchmark/reference strategy for framework validation. Simple EMA crossover + RSI.
- **Entry:** EMA(8) > EMA(21) + RSI confirmation
- **Risk:** stoploss=-3%, minimal_roi tiered

### 4.4 `RollingSmokeStrategy(IStrategy)`
- **Timeframe:** 5m, minimal smoke test strategy
- **Core:** EMA(10) > EMA(30) for long entry, inverse for short. Trivial exit on RSI extremes.
- **Purpose:** Quick validation that rolling backtest framework works correctly.

### 4.5 `WFOSmokeStrategy(IStrategy)`
- **Timeframe:** 5m, with `IntParameter`/`DecimalParameter` for WFO optimization
- **Core:** Parameterized EMA crossover for WFO smoke testing
- **Optimizable params:** `ema_fast` (5-20), `ema_slow` (20-50), `rsi_threshold` (55-75)
- **Purpose:** Validates WFO optimizer can find and apply parameters.

---

## 5. Configuration Schema

### Rolling Backtest Config (JSON)

```json
{
  "stake_currency": "USDT",
  "stake_amount": "unlimited",
  "dry_run_wallet": 10000,
  "trading_mode": "futures",
  "margin_mode": "isolated",
  "timeframe": "5m",
  "timerange": "20240101-20250101",
  "strategy": "StrategyName",
  "datadir": "/data/freqtrade_data",
  "dataformat_ohlcv": "feather",
  "max_open_trades": 10,
  "candle_type_def": "futures",

  "exchange": {
    "name": "binance",
    "pair_whitelist": ["BTC/USDT:USDT", ...],
    "pair_blacklist": []
  },

  "pairlists": [{"method": "StaticPairList"}],

  // Rolling-specific (optional, usually set via CLI)
  "enable_dynamic_pairlist": false,

  // Standard freqtrade fields
  "unfilledtimeout": { "entry": 10, "exit": 10 },
  "entry_pricing": { "price_side": "same", "use_order_book": true, "order_book_top": 1 },
  "exit_pricing": { "price_side": "same", "use_order_book": true, "order_book_top": 1 }
}
```

### WFO Config (JSON)
Same as rolling backtest config, with strategy having `IntParameter`/`DecimalParameter` with `optimize=True`.

### Available Config Presets

| Config File | Purpose | Pairs | Timerange |
|---|---|---|---|
| `smoke_test_config.json` | Quick CI/smoke test | 3 pairs | 30 days |
| `mini_test_config.json` | Small validation | ~10 pairs | 60 days |
| `benchmark_config.json` | Standard benchmark | ~30 pairs | 6 months |
| `benchmark_config_group2.json` | Alternate benchmark group | ~30 pairs | 6 months |
| `bench_2year_config.json` | Extended benchmark | ~30 pairs | 2 years |
| `fullmarket_config.json` | Full market (535+ pairs) | all USDT perpetuals | varies |
| `wfo_smoke_config.json` | WFO smoke test | 3 pairs | 120+ days |

---

## 6. Data Workflow

```
1. Manifest Generation
   download_binance_futures.py --manifest-only
   → /data/freqtrade_data/_meta/binance_futures_manifest.csv

2. Data Download
   download_binance_futures.py
   → /data/freqtrade_data/futures/binance/*.feather
   → /data/freqtrade_data/_meta/download_state.json (resumable)
   → /data/freqtrade_data/_meta/binance_futures_coverage.csv

3. Quality Checks
   check_data_quality.py    → gap/staleness report
   check_data_integrity.py  → cross-timeframe consistency

4. Pair Availability
   build_pair_availability.py
   → /data/freqtrade_data/_meta/pair_availability.csv

5. Exchange Cache (for offline backtest init)
   cache_exchange_data.py
   → /data/freqtrade_data/_meta/exchange_cache.json

6. Rolling Backtest
   run_rolling_backtest.py --config ... --strategy ...
   → output/rolling_backtest_result.json
   → user_data/backtest_results/*.zip (optional)

7. WFO
   run_wfo.py --config ... --strategy ...
   → user_data/wfo_results/wfo_*.json
```

---

## 7. Key Constraints and Notes

1. **Memory**: The framework is designed for large-scale backtests (500+ pairs × years of data). Auto-windowing prevents OOM by estimating `pairs × candles_per_day × 60 bytes × 2.5x indicator multiplier`.

2. **Trade Continuity**: Open trades survive window boundaries. The system extends data loading to cover entry points of open trades, ensuring continuous indicator calculation.

3. **Reproducibility**: `parity_tools.compute_trade_digest()` produces SHA-256 digests of canonical trade lists for bitwise-exact verification across runs.

4. **Futures Support**: Full support for Binance USDT-M futures including funding rates and mark prices. `wfo_fold.py` and `wfo_optimizer.py` both handle futures data loading.

5. **Signal Export**: Optional per-window signal recording with boundary deviation detection — useful for debugging indicator warm-up issues at window boundaries.

6. **Parallelism**: Thread-based only (TA-Lib releases GIL). Cross-pair post-processing (market ranking) runs serially in the main thread.

7. **Offline Mode**: Exchange market data cached to JSON, injected via monkey-patch to skip all network calls during `Backtesting.__init__()`.

8. **WFO Overfit Detection**: Uses IS vs OOS profit ratio, parameter stability (coefficient of variation), and efficiency ratio to quantify overfit risk with Chinese-language verdict descriptions.
