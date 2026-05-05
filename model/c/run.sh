#!/bin/bash
# C 模型一键编译运行脚本
# 无需本地安装 gcc —— 使用 Docker (nix-shell) 自动编译运行
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$SCRIPT_DIR/fma_model.c"

echo "=========================================="
echo "  FMA C Model (via Docker + nix + gcc)"
echo "=========================================="

if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker 未安装"
    echo "  安装 Docker Desktop 后重试"
    exit 1
fi

if ! docker image inspect efabless/openlane:latest &> /dev/null; then
    echo "[INFO] 拉取 Docker 镜像..."
    docker pull efabless/openlane:latest
fi

echo ""
echo "[编译 + 运行]"
echo ""

cat "$SRC" | docker run --rm -i --entrypoint bash efabless/openlane:latest -c "
cat > /tmp/fma_model.c && nix-shell -p gcc --run \"
cd /tmp && gcc -o fma_model fma_model.c -lm && ./fma_model
\""

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "  ✅ C 模型全部测试通过"
    echo "=========================================="
else
    echo "=========================================="
    echo "  ❌ 测试失败"
    echo "=========================================="
    exit $EXIT_CODE
fi
