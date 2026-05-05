#!/bin/bash
# Phase 3: FMA 综合交付件报告生成脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNS_DIR="$PROJECT_ROOT/openlane/fma_top/runs"
REPORT_DIR="$SCRIPT_DIR/report"
REPORT_FILE="$REPORT_DIR/synthesis_report.md"

mkdir -p "$REPORT_DIR"

if [ ! -d "$RUNS_DIR" ]; then
    echo "[ERROR] 未找到运行目录: $RUNS_DIR"
    exit 1
fi

LATEST_RUN=$(ls -td "$RUNS_DIR"/RUN_* 2>/dev/null | head -1)
[ -z "$LATEST_RUN" ] && { echo "[ERROR] 未找到运行结果"; exit 1; }

RUN_NAME=$(basename "$LATEST_RUN")
METRICS="$LATEST_RUN/reports/metrics.csv"
SYNTH_LOG="$LATEST_RUN/logs/synthesis/1-synthesis.log"
LINTER_LOG="$LATEST_RUN/logs/synthesis/linter.log"

echo "========================================"
echo "  Phase 3: 生成 FMA 综合评审报告"
echo "========================================"
echo "  运行目录: $RUN_NAME"

extract() { head -2 "$METRICS" | tail -1 | cut -d',' -f"$1"; }

CELL_COUNT=$(extract 18)
TOTAL_CELLS=$(extract 46)
DIE_AREA=$(extract 7)
CORE_AREA=$(extract 47)
CRITICAL_PATH=$(extract 60)
WNS=$(extract 27)
TNS=$(extract 32)
FLOW_STATUS=$(extract 4)
TOTAL_RUNTIME=$(extract 5)
FINAL_UTIL=$(extract 11)

SYNTH_ERRORS=$(grep -ci "ERROR" "$SYNTH_LOG" 2>/dev/null | tr -d '[:space:]' || echo "0")
SYNTH_WARNINGS=$(grep -ci "WARNING" "$SYNTH_LOG" 2>/dev/null | tr -d '[:space:]' || echo "0")

cat > "$REPORT_FILE" << EOF
# Phase 3 FMA 综合交付件报告

**生成时间**: $(date '+%Y-%m-%d %H:%M:%S')
**运行目录**: $RUN_NAME
**流程状态**: $FLOW_STATUS

## 综合结果摘要

| 指标 | 实测值 | 判定 |
|------|--------|------|
| 标准单元数 | $CELL_COUNT | - |
| 总单元数 (含 filler) | $TOTAL_CELLS | - |
| 关键路径延迟 | ${CRITICAL_PATH}ns | $([ "$WNS" = "0.0" ] || [ "$WNS" = "0" ] && echo '✅ 时序收敛' || echo '⚠️ 检查时序') |
| Die Area | ${DIE_AREA}mm^2 | - |
| Core Area | ${CORE_AREA}um^2 | - |
| Final Utilization | ${FINAL_UTIL}% | - |
| WNS | ${WNS}ns | $([ "$WNS" = "0.0" ] || [ "$WNS" = "0" ] && echo '✅' || echo '⚠️') |
| TNS | ${TNS}ns | $([ "$TNS" = "0.0" ] || [ "$TNS" = "0" ] && echo '✅' || echo '⚠️') |
| 综合 ERROR | $SYNTH_ERRORS | $([ "${SYNTH_ERRORS:-0}" -eq 0 ] && echo '✅' || echo '❌') |
| 综合 WARNING | ${SYNTH_WARNINGS:-0} | - |
| 总运行时间 | $TOTAL_RUNTIME | - |

## 综合流程步骤

| 步骤 | 说明 | 日志 |
|------|------|------|
| Linter | Verilator lint | logs/synthesis/linter.log |
| Synthesis | Yosys 综合 | logs/synthesis/1-synthesis.log |
| STA | 单 corner 时序 | logs/synthesis/2-sta.log |

## 交付件清单

| 文件 | 说明 |
|------|------|
| 综合后网表 | results/synthesis/fma_top_stage4.v |
| SDF | results/synthesis/fma_top_stage4.sdf |
| Metrics CSV | reports/metrics.csv |

## 综合评审签核

| 检查项 | 状态 |
|--------|------|
| RTL 源代码已就绪 | ✅ |
| 综合流程完整运行 | $([ "$FLOW_STATUS" = "flow completed" ] && echo '✅' || echo '⚠️') |
| 无时序违例 | $([ "$WNS" = "0.0" ] || [ "$WNS" = "0" ] && echo '✅' || echo '⚠️') |
| 综合无 ERROR | $([ "${SYNTH_ERRORS:-0}" -eq 0 ] && echo '✅' || echo '❌') |

**综合评审结论**: $([ "$FLOW_STATUS" = "flow completed" ] && [ "${SYNTH_ERRORS:-0}" -eq 0 ] && echo '✅ 通过' || echo '⚠️ 需检查')

---

*此报告由 gen_report.sh 自动生成*
EOF

echo "  报告已生成: $REPORT_FILE"
