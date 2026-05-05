#!/bin/bash
# FMA 芯片全流程开发 - 一键运行脚本
set -e
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "╔══════════════════════════════════════════╗"
echo "║  FMA 数字芯片全流程开发                   ║"
echo "║  Multi-Precision FMA + Sky130 + OpenLane  ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Phase 1: RTL
echo "━━━ Phase 1: RTL ━━━"
if [ -f "$PROJECT_ROOT/phase1_rtl/src/fma_top_stage4.v" ]; then
    LINES=$(wc -l < "$PROJECT_ROOT/phase1_rtl/src/fma_top_stage4.v")
    echo "  ✅ fma_top_stage4.v ($LINES 行)"
else
    echo "  ❌ RTL 不存在"; exit 1
fi
echo ""

# Phase 2: Simulation
echo "━━━ Phase 2: 仿真 ━━━"
if command -v iverilog &> /dev/null; then
    bash "$PROJECT_ROOT/phase2_sim/run_sim.sh"
else
    echo "  [SKIP] iverilog 未安装"
fi
echo ""

# Phase 3: Synthesis
echo "━━━ Phase 3: 综合 ━━━"
if command -v docker &> /dev/null; then
    bash "$PROJECT_ROOT/phase3_synthesis/run_synthesis.sh"
else
    echo "  [SKIP] Docker 未安装"
fi
echo ""

# Phase 4: PnR
echo "━━━ Phase 4: 布局布线 ━━━"
if command -v docker &> /dev/null; then
    bash "$PROJECT_ROOT/phase4_pnr/run_pnr.sh"
else
    echo "  [SKIP] Docker 未安装"
fi
echo ""

# Phase 5: Verification
echo "━━━ Phase 5: 物理验证 ━━━"
if [ -d "$PROJECT_ROOT/openlane/fma_top/runs" ]; then
    bash "$PROJECT_ROOT/phase5_verification/run_verify.sh"
else
    echo "  [SKIP] 未找到运行结果"
fi
echo ""

# Phase 6: GDS
echo "━━━ Phase 6: GDS 输出 ━━━"
if [ -d "$PROJECT_ROOT/openlane/fma_top/runs" ]; then
    bash "$PROJECT_ROOT/phase6_gds/run_gds.sh"
else
    echo "  [SKIP] 未找到运行结果"
fi
echo ""

echo "╔══════════════════════════════════════════╗"
echo "║  全流程执行完毕                           ║"
echo "╚══════════════════════════════════════════╝"
