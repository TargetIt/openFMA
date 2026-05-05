#!/bin/bash
# Phase 6: FMA GDS 输出检查脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PNR_RUNS="$PROJECT_ROOT/openlane/fma_top/runs"
REPORT_DIR="$SCRIPT_DIR/report"
REPORT_FILE="$REPORT_DIR/gds_report.md"

mkdir -p "$REPORT_DIR"

echo "========================================"
echo "  Phase 6: FMA GDS 输出"
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

FINAL="$LATEST_RUN/results/final"
ALL_PRESENT=1

check_file() {
    local file="$1"
    local label="$2"
    if [ -f "$file" ]; then
        local size=$(du -h "$file" | cut -f1)
        echo "  ✅ $label: $file ($size)"
        return 0
    else
        echo "  ❌ $label: 未找到"
        ALL_PRESENT=0
        return 1
    fi
}

echo ""
echo "输出文件检查："
echo "----------------------------------------"

check_file "$FINAL/gds/fma_top_stage4.gds" "GDS 版图"
check_file "$FINAL/lef/fma_top_stage4.lef" "LEF 库视图"
check_file "$FINAL/lib/fma_top_stage4.lib" "LIB 时序库"
check_file "$FINAL/def/fma_top_stage4.def" "DEF 设计交换"
check_file "$FINAL/verilog/gl/fma_top_stage4.v" "门级网表"
check_file "$FINAL/spi/lvs/fma_top_stage4.spice" "SPICE 网表"

# SDF
SDF_COUNT=$(find "$FINAL/sdf/" -name '*.sdf' 2>/dev/null | wc -l)
[ "$SDF_COUNT" -gt 0 ] && echo "  ✅ SDF 时序文件: $SDF_COUNT 个" || { echo "  ❌ SDF 文件未找到"; ALL_PRESENT=0; }

# SPEF
SPEF_COUNT=$(find "$FINAL/spef/" -name '*.spef' 2>/dev/null | wc -l)
[ "$SPEF_COUNT" -gt 0 ] && echo "  ✅ SPEF 寄生参数: $SPEF_COUNT 个" || echo "  ❌ SPEF 文件未找到"

cat > "$REPORT_FILE" << EOF
# Phase 6 FMA GDS 输出交付件报告

**生成时间**: $(date '+%Y-%m-%d %H:%M:%S')
**运行目录**: $RUN_NAME

## 主交付件

| 文件 | 大小 | 用途 |
|------|------|------|
| fma_top_stage4.gds | $(du -h "$FINAL/gds/fma_top_stage4.gds" 2>/dev/null | cut -f1 || echo 'N/A') | 流片主文件 |
| fma_top_stage4.lef | $(du -h "$FINAL/lef/fma_top_stage4.lef" 2>/dev/null | cut -f1 || echo 'N/A') | IP 集成接口 |
| fma_top_stage4.lib | $(du -h "$FINAL/lib/fma_top_stage4.lib" 2>/dev/null | cut -f1 || echo 'N/A') | 时序模型 |

## 辅助交付件

| 文件 | 说明 |
|------|------|
| fma_top_stage4.def | 设计交换格式 |
| fma_top_stage4.v (gl) | 门级网表 |
| fma_top_stage4.spice | SPICE 网表 |
| SDF ($SDF_COUNT 个) | 多 corner 标准延时 |
| SPEF ($SPEF_COUNT 个) | 寄生参数提取 |

## 签核结论

**$( [ "$ALL_PRESENT" -eq 1 ] && echo '✅ GDS 交付件完整' || echo '⚠️ 部分缺失')**

---

*此报告由 run_gds.sh 自动生成*
EOF

echo ""
echo "========================================"
echo "  GDS 报告已生成: $REPORT_FILE"
echo "========================================"
