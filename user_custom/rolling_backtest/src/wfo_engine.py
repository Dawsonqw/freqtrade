"""WFO Engine — orchestrate Walk-Forward Optimization across all folds."""
from __future__ import annotations

import gc
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from freqtrade.configuration import TimeRange
from freqtrade.optimize.backtesting import Backtesting

from .wfo_fold import FoldResult, WFOFoldRunner
from .wfo_optimizer import WFOOptimizer
from .wfo_report import generate_wfo_report
from .wfo_windowing import WFOFold, build_wfo_folds

logger = logging.getLogger(__name__)


class WFOEngine:
    """Walk-Forward Optimization engine.

    Orchestrates:
      1. Build IS/OOS fold windows
      2. For each fold: IS optimize → OOS validate
      3. Aggregate results + overfit analysis
    """

    def __init__(
        self,
        config: dict[str, Any],
        *,
        is_days: int = 90,
        oos_days: int = 30,
        step_days: int | None = None,
        n_epochs: int = 100,
        loss_function: str = "SharpeHyperOptLoss",
        spaces: list[str] | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        """
        Args:
            config: Freqtrade config dict (must have strategy, datadir, etc.)
            is_days: In-sample window length in days.
            oos_days: Out-of-sample window length in days.
            step_days: Slide step in days (default = oos_days for non-overlapping OOS).
            n_epochs: Optuna trials per fold.
            loss_function: Hyperopt loss function name.
            spaces: Parameter spaces to optimize (default: ["buy", "sell"]).
            output_dir: Directory for output reports. Defaults to config results dir.
        """
        self.config = config
        self.is_days = is_days
        self.oos_days = oos_days
        self.step_days = step_days
        self.n_epochs = n_epochs
        self.loss_function = loss_function
        self.spaces = spaces or ["buy", "sell"]

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(
                config.get("user_data_dir", "user_data")
            ) / "wfo_results"

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, timerange_str: str | None = None) -> dict[str, Any]:
        """Execute full WFO pipeline.

        Args:
            timerange_str: Optional timerange string (e.g. "20230101-20240101").
                          If None, uses config timerange.

        Returns:
            WFO report dict with folds, summary, and overfit analysis.
        """
        t0 = time.time()

        # Parse timerange
        if timerange_str:
            timerange = TimeRange.parse_timerange(timerange_str)
        elif self.config.get("timerange"):
            timerange = TimeRange.parse_timerange(self.config["timerange"])
        else:
            raise ValueError("No timerange specified. Use --timerange or config.")

        # Determine timeframe
        timeframe = self.config.get("timeframe", "5m")

        # Step 1: Build folds
        logger.info(
            "Building WFO folds: IS=%dd, OOS=%dd, step=%s",
            self.is_days, self.oos_days,
            f"{self.step_days}d" if self.step_days else "auto",
        )
        folds = build_wfo_folds(
            timerange,
            timeframe=timeframe,
            is_days=self.is_days,
            oos_days=self.oos_days,
            step_days=self.step_days,
        )
        logger.info("Generated %d WFO folds", len(folds))

        if not folds:
            raise RuntimeError("No folds generated — check timerange and window sizes.")

        # Log fold plan
        for f in folds:
            logger.info(
                "  Fold %d: IS [%s → %s] (%d days) | OOS [%s → %s] (%d days)",
                f.index,
                f.is_start.strftime("%Y-%m-%d"),
                f.is_end.strftime("%Y-%m-%d"),
                int(f.is_days),
                f.oos_start.strftime("%Y-%m-%d"),
                f.oos_end.strftime("%Y-%m-%d"),
                int(f.oos_days),
            )

        # Step 2: Initialize backtesting + optimizer (once, reused across folds)
        # Inject spaces into config so HyperOptableStrategy.in_space works
        self.config["spaces"] = self.spaces
        bt = Backtesting(self.config)
        # Bind strategy to backtesting (normally done inside backtest_one_strategy)
        bt._set_strategy(bt.strategylist[0])
        try:
            optimizer = WFOOptimizer(
                bt,
                n_epochs=self.n_epochs,
                loss_function=self.loss_function,
                spaces=self.spaces,
            )
            fold_runner = WFOFoldRunner(bt, optimizer)

            # Step 3: Execute folds sequentially
            fold_results: list[FoldResult] = []
            for fold in folds:
                logger.info(
                    "=" * 60 + "\n[WFO] Starting fold %d/%d\n" + "=" * 60,
                    fold.index + 1, len(folds),
                )

                fold_t0 = time.time()
                result = fold_runner.run_fold(fold)
                fold_elapsed = time.time() - fold_t0

                fold_results.append(result)
                logger.info(
                    "[WFO] Fold %d/%d %s in %.1fs | OOS profit: %.2f%%",
                    fold.index + 1, len(folds),
                    result.status,
                    fold_elapsed,
                    result.oos_metrics.get("profit_total", 0.0) * 100
                    if result.oos_metrics else 0.0,
                )

                # Force GC between folds
                gc.collect()

            # Step 4: Generate report
            strategy_name = self.config.get("strategy", "unknown")
            report_path = self.output_dir / f"wfo_{strategy_name}_{_ts_tag()}.json"
            report = generate_wfo_report(fold_results, output_path=report_path)

            # Add run metadata
            elapsed = time.time() - t0
            report["metadata"] = {
                "strategy": strategy_name,
                "timeframe": timeframe,
                "timerange": timerange_str or self.config.get("timerange", ""),
                "is_days": self.is_days,
                "oos_days": self.oos_days,
                "step_days": self.step_days or self.oos_days,
                "n_epochs": self.n_epochs,
                "loss_function": self.loss_function,
                "spaces": self.spaces,
                "total_elapsed_sec": round(elapsed, 1),
                "report_path": str(report_path),
            }

            # Overwrite report with metadata
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

            logger.info(
                "[WFO] Complete: %d folds in %.1fs — report: %s",
                len(folds), elapsed, report_path,
            )

            return report

        finally:
            bt.cleanup()


def _ts_tag() -> str:
    """Generate a timestamp tag for filenames."""
    from datetime import datetime, UTC
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
