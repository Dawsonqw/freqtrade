from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .window_metrics import WindowPerformance

logger = logging.getLogger(__name__)


@dataclass
class WindowComparison:
    """Cross-strategy comparison for a single window."""
    window_index: int
    window_start: str
    window_end: str
    strategy_metrics: dict[str, dict[str, Any]]  # strategy_name -> metrics dict
    best_strategy_profit: str = ""
    best_strategy_sharpe: str = ""
    best_strategy_win_rate: str = ""
    profit_spread: float = 0.0  # best - worst profit
    sharpe_spread: float = 0.0


@dataclass
class StrategyRanking:
    """Overall ranking entry for a strategy."""
    strategy: str
    total_profit_pct: float
    avg_sharpe: float
    avg_win_rate: float
    avg_max_drawdown: float
    windows_won_profit: int  # how many windows this strategy had best profit
    windows_won_sharpe: int
    consistency_score: float = 0.0  # lower std of profit across windows = more consistent


@dataclass
class StrategyWindowComparison:
    """Full comparison report across strategies and windows."""
    strategies: list[str]
    windows: list[WindowComparison]
    overall_ranking: list[StrategyRanking]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategies": self.strategies,
            "windows": [asdict(w) for w in self.windows],
            "overall_ranking": [asdict(r) for r in self.overall_ranking],
        }


def build_comparison_report(
    strategy_performances: dict[str, list[WindowPerformance]],
) -> StrategyWindowComparison:
    """Build a cross-strategy, per-window comparison report.

    Args:
        strategy_performances: {strategy_name: [WindowPerformance, ...]}

    Returns:
        StrategyWindowComparison with per-window and overall rankings.
    """
    strategies = list(strategy_performances.keys())
    if not strategies:
        return StrategyWindowComparison(strategies=[], windows=[], overall_ranking=[])

    # Determine number of windows from first strategy
    n_windows = len(next(iter(strategy_performances.values())))

    windows: list[WindowComparison] = []
    # Track wins per strategy
    wins_profit: dict[str, int] = {s: 0 for s in strategies}
    wins_sharpe: dict[str, int] = {s: 0 for s in strategies}

    for wi in range(n_windows):
        # Gather metrics for this window across strategies
        strategy_metrics: dict[str, dict[str, Any]] = {}
        for sname, perfs in strategy_performances.items():
            if wi < len(perfs):
                wp = perfs[wi]
                strategy_metrics[sname] = wp.to_dict()

        if not strategy_metrics:
            continue

        # Find best by different metrics
        best_profit_name = max(strategy_metrics, key=lambda s: strategy_metrics[s].get("total_profit_pct", -999))
        best_sharpe_name = max(strategy_metrics, key=lambda s: strategy_metrics[s].get("sharpe_ratio", -999))
        best_winrate_name = max(strategy_metrics, key=lambda s: strategy_metrics[s].get("win_rate", -999))

        wins_profit[best_profit_name] = wins_profit.get(best_profit_name, 0) + 1
        wins_sharpe[best_sharpe_name] = wins_sharpe.get(best_sharpe_name, 0) + 1

        profits = [strategy_metrics[s].get("total_profit_pct", 0) for s in strategy_metrics]
        sharpes = [strategy_metrics[s].get("sharpe_ratio", 0) for s in strategy_metrics]

        # Get window dates from first available strategy
        first_wp = next(iter(strategy_metrics.values()))
        wc = WindowComparison(
            window_index=wi,
            window_start=first_wp.get("window_start", ""),
            window_end=first_wp.get("window_end", ""),
            strategy_metrics=strategy_metrics,
            best_strategy_profit=best_profit_name,
            best_strategy_sharpe=best_sharpe_name,
            best_strategy_win_rate=best_winrate_name,
            profit_spread=max(profits) - min(profits) if profits else 0.0,
            sharpe_spread=max(sharpes) - min(sharpes) if sharpes else 0.0,
        )
        windows.append(wc)

    # Build overall ranking
    rankings: list[StrategyRanking] = []
    for sname, perfs in strategy_performances.items():
        profits = [wp.total_profit_pct for wp in perfs if wp.total_trades > 0]
        sharpes = [wp.sharpe_ratio for wp in perfs if wp.total_trades > 0]
        win_rates = [wp.win_rate for wp in perfs if wp.total_trades > 0]
        drawdowns = [wp.max_drawdown for wp in perfs if wp.total_trades > 0]

        rankings.append(StrategyRanking(
            strategy=sname,
            total_profit_pct=sum(wp.total_profit_pct for wp in perfs),
            avg_sharpe=float(np.mean(sharpes)) if sharpes else 0.0,
            avg_win_rate=float(np.mean(win_rates)) if win_rates else 0.0,
            avg_max_drawdown=float(np.mean(drawdowns)) if drawdowns else 0.0,
            windows_won_profit=wins_profit.get(sname, 0),
            windows_won_sharpe=wins_sharpe.get(sname, 0),
            consistency_score=float(np.std(profits)) if len(profits) > 1 else 0.0,
        ))

    # Sort by total profit descending
    rankings.sort(key=lambda r: r.total_profit_pct, reverse=True)

    return StrategyWindowComparison(
        strategies=strategies,
        windows=windows,
        overall_ranking=rankings,
    )


def format_comparison_table(report: StrategyWindowComparison) -> str:
    """Format comparison report as human-readable text table."""
    if not report.windows:
        return "No windows to compare."

    strategies = report.strategies
    col_width = max(len(s) for s in strategies) + 4
    val_width = max(col_width - 4, 6)

    lines = []
    lines.append(f"\n{'=' * (40 + col_width * len(strategies))}")
    lines.append(" Multi-Strategy Window Comparison")
    lines.append(f"{'=' * (40 + col_width * len(strategies))}")

    # Header row
    hdr = f" {'Window':>6} | {'Period':^23} |"
    for s in strategies:
        hdr += f" {s:^{col_width}} |"
    hdr += " Best"
    lines.append(hdr)
    lines.append(f"{'-' * (40 + col_width * len(strategies) + 6)}")

    # Profit comparison per window
    lines.append(" --- Profit % ---")
    for wc in report.windows:
        start_short = wc.window_start[:10]
        end_short = wc.window_end[:10]
        period = f"{start_short}→{end_short}"
        row = f" {f'Win {wc.window_index}':>6} | {period:^23} |"
        for s in strategies:
            val = wc.strategy_metrics.get(s, {}).get("total_profit_pct", 0)
            marker = " ★" if s == wc.best_strategy_profit else "  "
            row += f" {val:>+{val_width}.2f}{marker} |"
        row += f" {wc.best_strategy_profit}"
        lines.append(row)

    # Sharpe comparison per window
    lines.append(f"\n --- Sharpe Ratio ---")
    for wc in report.windows:
        start_short = wc.window_start[:10]
        end_short = wc.window_end[:10]
        period = f"{start_short}→{end_short}"
        row = f" {f'Win {wc.window_index}':>6} | {period:^23} |"
        for s in strategies:
            val = wc.strategy_metrics.get(s, {}).get("sharpe_ratio", 0)
            marker = " ★" if s == wc.best_strategy_sharpe else "  "
            row += f" {val:>+{val_width}.2f}{marker} |"
        row += f" {wc.best_strategy_sharpe}"
        lines.append(row)

    # Overall ranking
    lines.append(f"\n{'=' * (40 + col_width * len(strategies) + 6)}")
    lines.append(" Overall Ranking")
    lines.append(f"{'-' * 80}")
    lines.append(f" {'#':>2} | {'Strategy':<{col_width}} | {'TotalP%':>8} | {'AvgSharpe':>9} | "
                 f"{'AvgWin%':>7} | {'AvgDD%':>7} | {'WinsP':>5} | {'WinsS':>5} | {'Consist':>7}")
    lines.append(f"{'-' * 80}")
    for i, r in enumerate(report.overall_ranking):
        lines.append(
            f" {i+1:>2} | {r.strategy:<{col_width}} | {r.total_profit_pct:>+8.2f} | "
            f"{r.avg_sharpe:>9.2f} | {r.avg_win_rate * 100:>7.1f} | "
            f"{r.avg_max_drawdown * 100:>7.2f} | {r.windows_won_profit:>5} | "
            f"{r.windows_won_sharpe:>5} | {r.consistency_score:>7.2f}"
        )
    lines.append(f"{'=' * 80}")

    return "\n".join(lines)
