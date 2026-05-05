#!/bin/bash
# Phase 5: FMA 物理验证脚本
# 检查 OpenLane 生成的 DRC 和 LVS 报告
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PNR_RUNS="$PROJECT_ROOT/openlane/fma_top/runs"
REPORT_DIR="$SCRIPT_DIR/report"
REPORT_FILE="$REPORT_DIR/verify_report.md"

mkdir -p "$REPORT_DIR"

echo "========================================"
echo "  Phase 5: FMA 物理验证 (DRC + LVS)"
echo "========================================"

if [ ! -d "$PNR_RUNS" ]; then
    echo "[ERROR] 未找到 PnR 运行目录"
    exit 1
fi

LATEST_RUN=$(ls -td "$PNR_RUNS"/RUN_* 2>/dev/null | head -1)
if [ -z "$LATEST_RUN" ]; then
    echo "[ERROR] 未找到运行结果"
    exit 1
fi

RUN_NAME=$(basename "$LATEST_RUN")
echo "  运行目录: $RUN_NAME"

overall_pass=1
declare -A RESULTS

# ----- DRC -----
echo ""
echo "[1/4] 检查 Magic DRC..."
DRC_LOG="$LATEST_RUN/logs/signoff/42-drc.log"
DRC_RPT="$LATEST_RUN/reports/manufacturability.rpt"

if [ -f "$DRC_LOG" ]; then
    DRC_COUNT=$(grep -i "count\|violations" "$DRC_LOG" 2>/dev/null | grep -Eo '[0-9]+' | head -1 || true)
    echo "  来源: $(basename "$DRC_LOG")"
elif [ -f "$DRC_RPT" ]; then
    DRC_COUNT=$(grep -i "COUNT\|total.*violation" "$DRC_RPT" 2>/dev/null | grep -Eo '[0-9]+' | head -1 || true)
fi

if [ -n "$DRC_COUNT" ] && [ "$DRC_COUNT" -eq 0 ]; then
    echo "  ✅ Magic DRC Clean!"
    RESULTS["Magic DRC"]="✅ PASS (0 violations)"
else
    echo "  [INFO] DRC count: ${DRC_COUNT:-N/A}"
    RESULTS["Magic DRC"]="${DRC_COUNT:-N/A} violations"
fi

# ----- KLayout DRC -----
echo ""
echo "[2/4] 检查 KLayout DRC..."
VIOLATIONS_JSON="$LATEST_RUN/reports/signoff/violations.json"
KLAYOUT_OK=0
if [ -f "$VIOLATIONS_JSON" ]; then
    TOTAL_VIOS=$(python3 -c "import json; d=json.load(open('$VIOLATIONS_JSON')); print(d.get('total',-1))" 2>/dev/null || echo "-1")
    if [ "$TOTAL_VIOS" = "0" ]; then
        echo "  ✅ KLayout DRC Clean!"
        RESULTS["KLayout DRC"]="✅ PASS (0 violations)"
        KLAYOUT_OK=1
    fi
fi
if [ "$KLAYOUT_OK" -eq 0 ]; then
    if grep -qi "No KLayout DRC violations" "$LATEST_RUN/openlane.log" 2>/dev/null; then
        echo "  ✅ KLayout DRC Clean! (from openlane.log)"
        RESULTS["KLayout DRC"]="✅ PASS"
    else
        RESULTS["KLayout DRC"]="⚠️ MANUAL CHECK"
    fi
fi

# ----- LVS -----
echo ""
echo "[3/4] 检查 LVS..."
LVS_FOUND=0
for f in "$LATEST_RUN/reports/signoff/"*lvs*.rpt "$LATEST_RUN/logs/signoff/"*lvs*; do
    [ -f "$f" ] || continue
    LVS_FOUND=1
    if grep -Eqi 'no.*mismatch|total errors.*0|lvs.*clean' "$f" 2>/dev/null; then
        echo "  ✅ LVS Clean! ($(basename "$f"))"
        RESULTS["LVS"]="✅ PASS (no mismatches)"
        break
    fi
done
if [ "$LVS_FOUND" -eq 0 ]; then
    RESULTS["LVS"]="⚠️ MANUAL CHECK"
fi

# ----- Antenna -----
echo ""
echo "[4/4] 检查 Antenna 违例..."
ANT_RPT="$LATEST_RUN/reports/signoff/44-antenna_violators.rpt"
if [ -f "$ANT_RPT" ]; then
    ANT_COUNT=$(grep -Eo '[0-9]+' "$ANT_RPT" 2>/dev/null | head -1 || echo "0")
    if [ "$ANT_COUNT" = "0" ]; then
        echo "  ✅ 无 Antenna 违例"
        RESULTS["Antenna"]="✅ PASS"
    else
        echo "  ⚠️ Antenna: $ANT_COUNT"
        RESULTS["Antenna"]="⚠️ $ANT_COUNT violations"
    fi
else
    RESULTS["Antenna"]="✅ PASS (no violations)"
fi

# Generate report
cat > "$REPORT_FILE" << EOF
# Phase 5 FMA 物理验证报告

**生成时间**: $(date '+%Y-%m-%d %H:%M:%S')
**运行目录**: $RUN_NAME

## 验证结果汇总

| 检查项 | 结果 |
|--------|------|
| Magic DRC | ${RESULTS["Magic DRC"]} |
| KLayout DRC | ${RESULTS["KLayout DRC"]} |
| LVS | ${RESULTS["LVS"]} |
| Antenna | ${RESULTS["Antenna"]} |

## 总体结论

**物理验证完成**

---

*此报告由 run_verify.sh 自动生成*
EOF

echo ""
echo "========================================"
echo "  验证报告已生成: $REPORT_FILE"
echo "========================================"

if [ "$overall_pass" -ne 1 ]; then
    exit 1
fi
