#!/bin/bash
# Phase 4: FMA 布局布线脚本
# 使用 OpenLane (Docker) 运行完整 RTL -> GDS 流程
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DESIGN_DIR="$PROJECT_ROOT/openlane/fma_top"
VOLARE_DIR="$HOME/.volare"

echo "========================================"
echo "  Phase 4: FMA 布局布线 (OpenROAD via OpenLane)"
echo "========================================"

if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker 未安装。"
    exit 1
fi

if ! docker image inspect efabless/openlane:latest &> /dev/null; then
    echo "[INFO] OpenLane Docker 镜像不存在，正在拉取..."
    docker pull efabless/openlane:latest
fi

if [ ! -f "$DESIGN_DIR/config_pnr.tcl" ]; then
    echo "[ERROR] 配置文件不存在: $DESIGN_DIR/config_pnr.tcl"
    exit 1
fi

# Use PnR-optimized config
cp "$DESIGN_DIR/config_pnr.tcl" "$DESIGN_DIR/config.tcl"

echo "[1/2] 启动 OpenLane 完整流程 (RTL -> GDS)..."
docker run --rm \
    -v "$PROJECT_ROOT":/work \
    -v "$VOLARE_DIR":/root/.volare \
    -e PDK_ROOT=/root/.volare/volare/sky130/versions/c6d73a35f524070e85faff4a6a9eef49553ebc2b \
    -w /work/openlane/fma_top \
    efabless/openlane:latest \
    flow.tcl

echo ""
echo "========================================"
echo "  布局布线完成！"
echo "  报告位于: openlane/fma_top/runs/ 目录下"
echo "========================================"
echo ""
echo "配置参数："
echo "  - CLOCK_PERIOD: 10.0 ns (100 MHz)"
echo "  - FP_CORE_UTIL: 40"
echo "  - PL_TARGET_DENSITY: 0.5"
