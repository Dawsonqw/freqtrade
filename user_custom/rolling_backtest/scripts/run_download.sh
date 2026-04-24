#!/usr/bin/env bash
# ============================================================================
# Binance Futures 历史数据下载启动脚本
#
# 环境要求:
#   - /root/workspace/freqtrade/.venv/bin/python3 (Python 3.11)
#   - freqtrade 已安装在 venv 中
#   - Binance fapi 可直连（无需代理）
#   - 数据存储到 /data/freqtrade_data/（20G 挂载盘）
#
# ============================================================================
# 用法:
#
#   前台交互运行（可看实时输出，Ctrl+C 安全中断）:
#     ./run_download.sh              # 首次全量下载（并行，最快，约400+ pairs × 6 tf × 5年）
#     ./run_download.sh full         # 同上
#     ./run_download.sh resume       # 恢复中断的下载（自动跳过已完成任务）
#     ./run_download.sh prepend      # 增量补历史数据（往已有数据前面补，自动关闭并行）
#     ./run_download.sh retry        # 重试之前失败的任务
#     ./run_download.sh manifest     # 只拉交易对列表，不下载数据
#     ./run_download.sh coverage     # 只统计已有数据的时间覆盖范围
#     ./run_download.sh smoke        # 烟雾测试（3个pair，只下5m，约1分钟）
#     ./run_download.sh status       # 查看当前下载进度（不启动下载）
#
#   后台运行（推荐全量下载时使用，预计数小时）:
#     nohup ./run_download.sh full > /data/freqtrade_data/_logs/full_download.log 2>&1 &
#
#   后台运行后查看进度:
#     ./run_download.sh status                                   # 看进度摘要
#     tail -f /data/freqtrade_data/_logs/full_download.log       # 看实时日志
#     du -sh /data/freqtrade_data/futures/                       # 看已下载数据大小
#
#   断点续传:
#     任何时候 Ctrl+C 或 kill 都会安全保存进度
#     再次运行 ./run_download.sh 即自动从断点恢复
#
#   全量下载完成后日常增量更新:
#     ./run_download.sh full         # 会自动跳过已有数据，只下新K线
#
# 数据目录结构:
#   /data/freqtrade_data/
#     futures/                       # feather 格式 K 线文件
#       BTC_USDT_USDT-5m-futures.feather
#       BTC_USDT_USDT-1h-futures.feather
#       ...
#     _meta/
#       binance_futures_manifest.csv # 交易对元数据
#       binance_futures_coverage.csv # 数据覆盖统计
#       download_state.json          # 下载进度（断点续传依赖此文件）
#       download_report.json         # 完成后的汇总报告
#     _logs/
#       download_20260424_105943.log # 按时间戳命名的日志
#
# ============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python3"
DOWNLOAD_SCRIPT="$SCRIPT_DIR/download_binance_futures.py"

DATA_DIR="/data/freqtrade_data"
META_DIR="/data/freqtrade_data/_meta"
STATE_FILE="$META_DIR/download_state.json"
LOG_DIR="/data/freqtrade_data/_logs"

# ---------------------------------------------------------------------------
# 前置检查
# ---------------------------------------------------------------------------
if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "ERROR: venv python not found at $VENV_PYTHON"
    echo "  cd $REPO_ROOT && python3 -m venv .venv && .venv/bin/pip install -e ."
    exit 1
fi

if ! "$VENV_PYTHON" -c "import freqtrade" 2>/dev/null; then
    echo "ERROR: freqtrade not installed in venv"
    echo "  cd $REPO_ROOT && .venv/bin/pip install -e ."
    exit 1
fi

mkdir -p "$DATA_DIR" "$META_DIR" "$LOG_DIR"

# ---------------------------------------------------------------------------
# 日志文件（按日期）
# ---------------------------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/download_${TIMESTAMP}.log"

# ---------------------------------------------------------------------------
# 运行模式
# ---------------------------------------------------------------------------
MODE="${1:-full}"

case "$MODE" in
    full|resume)
        echo "=== 全量/恢复下载模式 ==="
        echo "  数据目录: $DATA_DIR"
        echo "  日志文件: $LOG_FILE"
        echo "  并行模式: ON (prepend=OFF)"
        echo ""
        ARGS=(
            --data-dir "$DATA_DIR"
            --meta-dir "$META_DIR"
        )
        ;;

    prepend)
        echo "=== 增量补数据模式 (prepend) ==="
        echo "  注意: prepend 模式下并行下载自动关闭"
        echo ""
        ARGS=(
            --data-dir "$DATA_DIR"
            --meta-dir "$META_DIR"
            --prepend
        )
        ;;

    retry)
        echo "=== 重试失败任务 ==="
        if [[ -f "$STATE_FILE" ]]; then
            FAILED=$(python3 -c "
import json
s = json.load(open('$STATE_FILE'))
print(s.get('failed_count', 0))
" 2>/dev/null || echo "?")
            echo "  当前失败任务数: $FAILED"
        fi
        echo ""
        ARGS=(
            --data-dir "$DATA_DIR"
            --meta-dir "$META_DIR"
            --retry-failed
        )
        ;;

    manifest)
        echo "=== 仅拉取交易对列表 ==="
        ARGS=(
            --data-dir "$DATA_DIR"
            --meta-dir "$META_DIR"
            --manifest-only
        )
        ;;

    coverage)
        echo "=== 仅统计数据覆盖范围 ==="
        ARGS=(
            --data-dir "$DATA_DIR"
            --meta-dir "$META_DIR"
            --coverage-only
        )
        ;;

    smoke)
        echo "=== 烟雾测试（3 pairs, 5m only）==="
        ARGS=(
            --data-dir "$DATA_DIR"
            --meta-dir "$META_DIR"
            --max-pairs 3
            --timeframes 5m
            --reset-state
        )
        ;;

    status)
        echo "=== 下载进度 ==="
        if [[ ! -f "$STATE_FILE" ]]; then
            echo "  尚未开始下载（状态文件不存在）"
            exit 0
        fi
        python3 -c "
import json, sys
s = json.load(open('$STATE_FILE'))
total = s.get('total_tasks', 0)
done  = s.get('completed_count', 0)
fail  = s.get('failed_count', 0)
pct   = done / total * 100 if total > 0 else 0
print(f'  总任务数:   {total}')
print(f'  已完成:     {done} ({pct:.1f}%)')
print(f'  失败:       {fail}')
print(f'  待处理:     {total - done - fail}')
print(f'  开始时间:   {s.get(\"started_at\", \"N/A\")}')
print(f'  最后更新:   {s.get(\"updated_at\", \"N/A\")}')
if fail > 0:
    print()
    print('  失败详情（前10条）:')
    for i, (k, v) in enumerate(s.get('failed', {}).items()):
        if i >= 10:
            print(f'  ... 还有 {fail - 10} 条')
            break
        print(f'    {k}: {v[:80]}')
"
        exit 0
        ;;

    help|-h|--help)
        # 提取脚本头部注释作为帮助信息（第2行到 set -euo 之前）
        sed -n '2,/^set -euo/{ /^set -euo/d; s/^# \{0,1\}//; p }' "$0"
        exit 0
        ;;

    *)
        echo "未知模式: $MODE"
        echo ""
        echo "用法: $0 {full|resume|prepend|retry|manifest|coverage|smoke|status|help}"
        echo ""
        echo "  full/resume  全量下载或恢复中断（默认）"
        echo "  prepend      增量补历史数据"
        echo "  retry        重试失败任务"
        echo "  manifest     只拉取交易对列表"
        echo "  coverage     统计数据覆盖范围"
        echo "  smoke        烟雾测试（3 pairs, 5m）"
        echo "  status       查看下载进度"
        echo "  help         显示完整帮助"
        echo ""
        echo "后台运行:"
        echo "  nohup $0 full > /data/freqtrade_data/_logs/full_download.log 2>&1 &"
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# 执行下载
# ---------------------------------------------------------------------------
echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "命令: $VENV_PYTHON $DOWNLOAD_SCRIPT ${ARGS[*]}"
echo "日志: $LOG_FILE"
echo "---"
echo "按 Ctrl+C 可安全中断（会保存进度，下次自动恢复）"
echo ""

# 同时输出到终端和日志文件
cd "$REPO_ROOT"
"$VENV_PYTHON" "$DOWNLOAD_SCRIPT" "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "---"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "退出码: $EXIT_CODE"
echo "日志: $LOG_FILE"

if [[ $EXIT_CODE -eq 2 ]]; then
    echo ""
    echo "⚠ 部分任务失败，运行以下命令重试:"
    echo "  $0 retry"
fi

exit $EXIT_CODE
