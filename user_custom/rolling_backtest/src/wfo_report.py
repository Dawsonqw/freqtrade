"""WFO Report — aggregate fold results, detect overfit, generate summary."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .wfo_fold import FoldResult

logger = logging.getLogger(__name__)


@dataclass
class WFOSummary:
    """Aggregated WFO results."""
    total_folds: int
    ok_folds: int
    failed_folds: int
    is_total_profit: float
    oos_total_profit: float
    oos_total_trades: int
    avg_efficiency_ratio: float | None
    overfit_score: float  # 0 = perfect, 1 = total overfit
    param_stability: float  # 0 = unstable, 1 = perfectly stable


def generate_wfo_report(
    folds: list[FoldResult],
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Generate WFO summary report from fold results.

    Returns a dict with folds detail + summary + overfit analysis.
    """
    ok_folds = [f for f in folds if f.status == "ok"]

    # IS/OOS aggregation
    is_total_profit = sum(f.is_metrics.get("profit_total", 0.0) for f in ok_folds)
    oos_total_profit = sum(f.oos_metrics.get("profit_total", 0.0) for f in ok_folds)
    oos_total_trades = sum(f.oos_metrics.get("trade_count", 0) for f in ok_folds)

    # Efficiency ratios
    efficiencies = [f.efficiency_ratio for f in ok_folds if f.efficiency_ratio is not None]
    avg_efficiency = sum(efficiencies) / len(efficiencies) if efficiencies else None

    # Overfit score: 1 - (OOS_profit / IS_profit), clamped to [0, 1]
    if is_total_profit > 0:
        overfit_score = max(0.0, min(1.0, 1.0 - (oos_total_profit / is_total_profit)))
    elif is_total_profit == 0:
        overfit_score = 0.5  # indeterminate
    else:
        # IS negative — if OOS also negative, not necessarily overfit
        overfit_score = 0.5

    # Parameter stability: how consistent are best_params across folds
    param_stability = _compute_param_stability(ok_folds)

    report = {
        "folds": [asdict(f) for f in folds],
        "summary": {
            "total_folds": len(folds),
            "ok_folds": len(ok_folds),
            "failed_folds": len(folds) - len(ok_folds),
            "is_total_profit_pct": round(is_total_profit * 100, 4),
            "oos_total_profit_pct": round(oos_total_profit * 100, 4),
            "oos_total_trades": oos_total_trades,
            "avg_efficiency_ratio": round(avg_efficiency, 4) if avg_efficiency else None,
            "overfit_score": round(overfit_score, 4),
            "param_stability": round(param_stability, 4),
        },
        "overfit_analysis": _overfit_analysis(ok_folds, overfit_score, avg_efficiency),
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info("WFO report saved to %s", output_path)

    # Print console summary
    _print_wfo_summary(report)

    return report


def _compute_param_stability(folds: list[FoldResult]) -> float:
    """Compute parameter stability across folds (0=unstable, 1=stable).

    Uses coefficient of variation for numeric params.
    """
    if len(folds) < 2:
        return 1.0

    all_params = [f.best_params for f in folds if f.best_params]
    if not all_params:
        return 0.0

    # Get common numeric keys
    keys = set(all_params[0].keys())
    for p in all_params[1:]:
        keys &= set(p.keys())

    if not keys:
        return 0.0

    stabilities = []
    for key in keys:
        values = [p[key] for p in all_params if isinstance(p.get(key), (int, float))]
        if len(values) < 2:
            continue
        mean = sum(values) / len(values)
        if mean == 0:
            continue
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        cv = (variance ** 0.5) / abs(mean)  # coefficient of variation
        # Convert CV to stability: CV=0 → 1.0, CV>=1 → 0.0
        stabilities.append(max(0.0, 1.0 - cv))

    return sum(stabilities) / len(stabilities) if stabilities else 0.0


def _overfit_analysis(
    folds: list[FoldResult],
    overfit_score: float,
    avg_efficiency: float | None,
) -> dict[str, Any]:
    """Generate human-readable overfit analysis."""
    analysis: dict[str, Any] = {}

    if overfit_score < 0.3:
        analysis["verdict"] = "LOW_OVERFIT"
        analysis["description"] = "策略 OOS 表现接近 IS，过拟合风险低。"
    elif overfit_score < 0.6:
        analysis["verdict"] = "MODERATE_OVERFIT"
        analysis["description"] = (
            "策略 OOS 表现明显弱于 IS，存在一定过拟合。"
            "建议增大 IS 窗口或简化参数空间。"
        )
    else:
        analysis["verdict"] = "HIGH_OVERFIT"
        analysis["description"] = (
            "策略 OOS 表现远弱于 IS，严重过拟合。"
            "建议减少优化参数数量、使用更长 IS 窗口、或换用更鲁棒的 loss function。"
        )

    # Per-fold IS vs OOS comparison
    analysis["fold_comparison"] = []
    for f in folds:
        is_p = f.is_metrics.get("profit_total", 0.0) * 100
        oos_p = f.oos_metrics.get("profit_total", 0.0) * 100
        analysis["fold_comparison"].append({
            "fold": f.fold_index,
            "is_profit_pct": round(is_p, 2),
            "oos_profit_pct": round(oos_p, 2),
            "efficiency": round(f.efficiency_ratio, 4) if f.efficiency_ratio else None,
        })

    return analysis


def _print_wfo_summary(report: dict) -> None:
    """Print compact WFO summary to console."""
    s = report["summary"]
    a = report["overfit_analysis"]

    lines = [
        "",
        "=" * 80,
        " Walk-Forward Optimization Summary",
        "=" * 80,
        f" Folds: {s['ok_folds']}/{s['total_folds']} ok | "
        f"IS Profit: {s['is_total_profit_pct']:.2f}% | "
        f"OOS Profit: {s['oos_total_profit_pct']:.2f}% | "
        f"OOS Trades: {s['oos_total_trades']}",
        f" Avg Efficiency: {s['avg_efficiency_ratio']:.4f}"
        if s["avg_efficiency_ratio"]
        else " Avg Efficiency: N/A",
        f" Overfit Score: {s['overfit_score']:.4f} | "
        f"Param Stability: {s['param_stability']:.4f}",
        f" Verdict: {a['verdict']}",
        "-" * 80,
        f" {'Fold':>4} | {'IS Profit':>10} | {'OOS Profit':>10} | {'Efficiency':>10}",
        "-" * 80,
    ]

    for fc in a.get("fold_comparison", []):
        eff_str = f"{fc['efficiency']:.4f}" if fc["efficiency"] is not None else "N/A"
        lines.append(
            f" {fc['fold']:>4} | {fc['is_profit_pct']:>9.2f}% | "
            f"{fc['oos_profit_pct']:>9.2f}% | {eff_str:>10}"
        )

    lines.append("=" * 80)
    print("\n".join(lines))
