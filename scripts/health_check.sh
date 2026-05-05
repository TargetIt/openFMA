#!/bin/bash
# FMA Project Health Check
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

pass_count=0
fail_count=0

check() {
    if [ -e "$1" ]; then
        echo "  [PASS] $2"
        pass_count=$((pass_count + 1))
    else
        echo "  [FAIL] $2"
        fail_count=$((fail_count + 1))
    fi
}

echo "========================================"
echo "  openFMA Health Check"
echo "========================================"

echo "[1/3] Required structure..."
check "$PROJECT_ROOT/phase1_rtl/src/fma_top_stage4.v" "RTL source"
check "$PROJECT_ROOT/phase2_sim/tb/tb_fma_top_stage4.v" "testbench"
check "$PROJECT_ROOT/phase2_sim/run_sim.sh" "sim script"
check "$PROJECT_ROOT/openlane/fma_top/config.tcl" "OpenLane config"
check "$PROJECT_ROOT/phase3_synthesis/run_synthesis.sh" "synthesis script"
check "$PROJECT_ROOT/phase4_pnr/run_pnr.sh" "PnR script"
check "$PROJECT_ROOT/phase5_verification/run_verify.sh" "verify script"
check "$PROJECT_ROOT/phase6_gds/run_gds.sh" "GDS script"
check "$PROJECT_ROOT/run_all.sh" "run_all.sh"

echo ""
echo "[2/3] Config keys..."
for field in DESIGN_NAME PDK CLOCK_PORT; do
    if grep -q "\"$field\"" "$PROJECT_ROOT/openlane/fma_top/config.tcl" 2>/dev/null || \
       grep -q "set.*$field" "$PROJECT_ROOT/openlane/fma_top/config.tcl" 2>/dev/null; then
        echo "  [PASS] config.tcl has '$field'"
        pass_count=$((pass_count + 1))
    else
        echo "  [FAIL] config.tcl missing '$field'"
        fail_count=$((fail_count + 1))
    fi
done

echo ""
echo "[3/3] Simulation smoke test..."
if command -v iverilog >/dev/null 2>&1; then
    if bash "$PROJECT_ROOT/phase2_sim/run_sim.sh" >/dev/null 2>&1; then
        echo "  [PASS] simulation passed"
        pass_count=$((pass_count + 1))
    else
        echo "  [FAIL] simulation failed"
        fail_count=$((fail_count + 1))
    fi
else
    echo "  [WARN] iverilog not installed"
fi

echo ""
echo "========================================"
echo "  Summary: PASS=$pass_count FAIL=$fail_count"
echo "========================================"

[ "$fail_count" -ne 0 ] && exit 1
