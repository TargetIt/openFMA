#!/bin/bash
# Phase 4: FMA PnR 交付件报告生成脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNS_DIR="$PROJECT_ROOT/openlane/fma_top/runs"
REPORT_DIR="$SCRIPT_DIR/report"
REPORT_FILE="$REPORT_DIR/pnr_report.md"

mkdir -p "$REPORT_DIR"

if [ ! -d "$RUNS_DIR" ]; then
    echo "[ERROR] 未找到运行目录: $RUNS_DIR"
    exit 1
fi

LATEST_RUN=$(ls -td "$RUNS_DIR"/RUN_* 2>/dev/null | head -1)
[ -z "$LATEST_RUN" ] && { echo "[ERROR] 未找到运行结果"; exit 1; }

RUN_NAME=$(basename "$LATEST_RUN")
METRICS="$LATEST_RUN/reports/metrics.csv"
WARNINGS_LOG="$LATEST_RUN/warnings.log"

echo "========================================"
echo "  Phase 4: 生成 FMA PnR 评审报告"
echo "========================================"
echo "  运行目录: $RUN_NAME"

extract() { head -2 "$METRICS" | tail -1 | cut -d',' -f"$1"; }

CELL_COUNT=$(extract 18)
TOTAL_CELLS=$(extract 46)
DIE_AREA=$(extract 7)
CORE_AREA=$(extract 47)
FINAL_UTIL=$(extract 11)
CRITICAL_PATH=$(extract 60)
CLOCK_PERIOD=$(extract 62)
WNS=$(extract 27)
TNS=$(extract 32)
ROUTING_VIOS=$(extract 20)
SHORT_VIOS=$(extract 21)
METSPC_VIOS=$(extract 22)
MAGIC_VIOS=$(extract 25)
LVS_ERRS=$(extract 28)
KLAYOUT_VIOS=$(extract 29)
WIRE_LENGTH=$(extract 24)
FLOW_STATUS=$(extract 4)
ROUTED_RUNTIME=$(extract 6)
PEAK_MEMORY=$(extract 12)

WARNING_COUNT=$(wc -l < "$WARNINGS_LOG" 2>/dev/null || echo 0)

cat > "$REPORT_FILE" << EOF
# Phase 4 FMA 布局布线交付件报告

**生成时间**: $(date '+%Y-%m-%d %H:%M:%S')
**运行目录**: $RUN_NAME
**流程状态**: $FLOW_STATUS

## PnR 结果摘要

### 面积与利用率

| 指标 | 实测值 | 判定 |
|------|--------|------|
| Die Area | ${DIE_AREA}mm^2 | - |
| Core Area | ${CORE_AREA}um^2 | - |
| Final Utilization | ${FINAL_UTIL}% | - |
| 综合单元数 | $CELL_COUNT | - |
| 总单元数 | $TOTAL_CELLS | - |

### 时序收敛

| 指标 | 实测值 | 约束 | 判定 |
|------|--------|------|------|
| 时钟周期 | ${CLOCK_PERIOD}ns | 10ns | - |
| 关键路径 | ${CRITICAL_PATH}ns | < 10ns | $([ "$WNS" = "0.0" ] || [ "$WNS" = "0" ] && echo '✅' || echo '⚠️') |
| WNS | ${WNS}ns | >= 0 | $([ "$WNS" = "0.0" ] || [ "$WNS" = "0" ] && echo '✅' || echo '⚠️') |
| TNS | ${TNS}ns | 0 | $([ "$TNS" = "0.0" ] || [ "$TNS" = "0" ] && echo '✅' || echo '⚠️') |

### 布线质量

| 指标 | 实测值 | 目标 | 判定 |
|------|--------|------|------|
| 布线违例 | $ROUTING_VIOS | 0 | $([ "$ROUTING_VIOS" = "0" ] && echo '✅' || echo '❌') |
| Short 违例 | $SHORT_VIOS | 0 | $([ "$SHORT_VIOS" = "0" ] && echo '✅' || echo '❌') |
| MetSpc 违例 | $METSPC_VIOS | 0 | $([ "$METSPC_VIOS" = "0" ] && echo '✅' || echo '❌') |
| 总走线长度 | ${WIRE_LENGTH}um | - | - |

### 物理验证

| 指标 | 实测值 | 判定 |
|------|--------|------|
| Magic DRC | $MAGIC_VIOS | $([ "$MAGIC_VIOS" = "0" ] && echo '✅' || echo '❌') |
| KLayout DRC | $KLAYOUT_VIOS | $([ "$KLAYOUT_VIOS" = "0" ] && echo '✅' || echo '❌') |
| LVS | $LVS_ERRS | $([ "$LVS_ERRS" = "0" ] && echo '✅' || echo '❌') |

### 资源消耗

| 指标 | 值 |
|------|-----|
| Routed 耗时 | $ROUTED_RUNTIME |
| 峰值内存 | ${PEAK_MEMORY}MB |

## PnR 评审签核

| 检查项 | 状态 |
|--------|------|
| 全流程完成 | $([ "$FLOW_STATUS" = "flow completed" ] && echo '✅' || echo '⚠️') |
| 布线无违例 | $([ "$ROUTING_VIOS" = "0" ] && echo '✅' || echo '❌') |
| DRC clean | $([ "$MAGIC_VIOS" = "0" ] && echo '✅' || echo '❌') |
| LVS clean | $([ "$LVS_ERRS" = "0" ] && echo '✅' || echo '❌') |
| 时序收敛 | $([ "$WNS" = "0.0" ] || [ "$WNS" = "0" ] && echo '✅' || echo '⚠️') |

**PnR 评审结论**: $([ "$FLOW_STATUS" = "flow completed" ] && [ "$MAGIC_VIOS" = "0" ] && [ "$LVS_ERRS" = "0" ] && echo '✅ 通过' || echo '⚠️ 需检查')

---

*此报告由 gen_report.sh 自动生成*
EOF

echo "  报告已生成: $REPORT_FILE"
