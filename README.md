# openFMA

多精度 FMA（Fused Multiply-Add）运算单元，分阶段迭代实现。

## 项目结构

```
openFMA/
├── doc/                    # 各阶段设计文档
│   ├── stage0_design.md    # 阶段 0: FP32 FMA 基准设计
│   ├── stage1_design.md    # 阶段 1: FP16 FMA 支持
│   ├── stage2_design.md    # 阶段 2: FP8×4 FMA 支持
│   ├── stage3_design.md    # 阶段 3: 混合精度累加模式
│   └── stage4_design.md    # 阶段 4: INT8 MAD 支持
├── rtl/                    # 各阶段独立 Verilog 代码
│   ├── stage0/fma_fp32_stage0.v
│   ├── stage1/fma_fp16_stage1.v
│   ├── stage2/fma_fp8x4_stage2.v
│   ├── stage3/fma_acc_stage3.v
│   └── stage4/fma_top_stage4.v
├── tb/                     # 各阶段测试平台
│   ├── tb_fma_fp32_stage0.v
│   ├── tb_fma_fp16_stage1.v
│   ├── tb_fma_fp8x4_stage2.v
│   ├── tb_fma_acc_stage3.v
│   └── tb_fma_top_stage4.v
└── or.md                   # 需求规格文档
```

## 模式编码

| `mode[2:0]` | 操作模式 |
|-------------|---------|
| `3'b000` | FP32 FMA |
| `3'b001` | FP16 FMA |
| `3'b010` | FP8×4 FMA |
| `3'b011` | INT8 MAD |
| `3'b100` | FP16×FP16 → FP32 Acc |
| `3'b101` | FP8×4 → FP32 Acc |

## 仿真

使用 Icarus Verilog 进行仿真：

```bash
# 阶段 0: FP32 FMA
iverilog -o sim rtl/stage0/fma_fp32_stage0.v tb/tb_fma_fp32_stage0.v && vvp sim

# 阶段 4: 完整 FMA Top
iverilog -o sim rtl/stage4/fma_top_stage4.v tb/tb_fma_top_stage4.v && vvp sim
```