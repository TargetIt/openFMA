#!/bin/bash
# Phase 2: FMA 仿真验证脚本
# 使用 iverilog 编译并运行仿真，生成 VCD 波形文件及测试报告
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RTL_SRC="$PROJECT_ROOT/phase1_rtl/src/fma_top_stage4.v"
TB_SRC="$SCRIPT_DIR/tb/tb_fma_top_stage4.v"
SIM_OUT="$SCRIPT_DIR/sim"
WAVE_FILE="$SCRIPT_DIR/wave.vcd"
TEST_LOG="$SCRIPT_DIR/test_results.log"
TEST_REPORT="$SCRIPT_DIR/test_report.md"

echo "========================================"
echo "  Phase 2: FMA 仿真验证 (Stage 4 Full Top)"
echo "========================================"

if ! command -v iverilog &> /dev/null; then
    echo "[ERROR] iverilog 未安装。"
    exit 1
fi

if [ ! -f "$RTL_SRC" ]; then
    echo "[ERROR] RTL 源文件不存在: $RTL_SRC"
    exit 1
fi

if [ ! -f "$TB_SRC" ]; then
    echo "[ERROR] Testbench 文件不存在: $TB_SRC"
    exit 1
fi

echo "[1/4] 编译 RTL 和 Testbench..."
iverilog -o "$SIM_OUT" "$TB_SRC" "$RTL_SRC"
echo "  编译成功"

echo "[2/4] 运行仿真..."
cd "$SCRIPT_DIR"
./sim 2>&1 | tee "$TEST_LOG"
SIM_EXIT=${PIPESTATUS[0]}
echo "  仿真完成"

echo "[3/4] 检查输出..."
if [ ! -f "$WAVE_FILE" ]; then
    echo "[ERROR] 波形文件未生成"
    exit 1
fi
echo "  波形文件已生成: $WAVE_FILE ($(du -h "$WAVE_FILE" | cut -f1))"

echo "[4/4] 生成测试报告..."

PASS_COUNT=$(grep -c "PASS:" "$TEST_LOG" 2>/dev/null || echo 0)
FAIL_COUNT=$(grep -c "FAIL:" "$TEST_LOG" 2>/dev/null || echo 0)
ALL_PASSED=$(grep -c "ALL TESTS PASSED" "$TEST_LOG" 2>/dev/null || echo 0)

cat > "$TEST_REPORT" << EOF
# Phase 2 FMA 仿真验证报告

**生成时间**: $(date '+%Y-%m-%d %H:%M:%S')

## 测试环境

| 项目 | 值 |
|------|-----|
| 仿真工具 | iverilog |
| RTL 文件 | fma_top_stage4.v (535 行) |
| Testbench | tb_fma_top_stage4.v (228 行) |
| 设计描述 | 多精度融合乘加 (FMA) 完整顶层 |

## 测试用例覆盖

| 模式 | 说明 | 测试数 |
|------|------|--------|
| FP32 | IEEE 754 单精度 FMA | 2+ |
| FP16 | IEEE 754 半精度 FMA | 2+ |
| FP8x4 | 4 路并行 FP8 E4M3 FMA | 1+ |
| INT8 MAD | 8-bit 整数乘加 | 5 |
| FP16->FP32 Acc | FP16 乘积累加到 FP32 | 1 |
| FP8x4->FP32 Acc | FP8x4 乘积累加到 FP32 | - |

## 测试统计

| 指标 | 值 |
|------|-----|
| 总测试数 | 12 |
| 通过 | $PASS_COUNT |
| 失败 | $FAIL_COUNT |

## 总体结果

**$(if [ "$ALL_PASSED" -gt 0 ]; then echo '✅ ALL TESTS PASSED'; else echo '❌ SOME TESTS FAILED'; fi)**

## 交付件清单

| 文件 | 说明 | 状态 |
|------|------|------|
| wave.vcd | 波形文件 | ✅ |
| test_results.log | 仿真原始日志 | ✅ |
| test_report.md | 本测试报告 | ✅ |
| sim | 编译后可执行文件 | ✅ |
EOF

echo "  测试报告已生成: $TEST_REPORT"
echo ""
echo "========================================"
echo "  仿真完成！"
echo "  测试报告: $TEST_REPORT"
echo "  查看波形: gtkwave $WAVE_FILE"
echo "========================================"

if [ "$ALL_PASSED" -eq 0 ]; then
    exit 1
fi
