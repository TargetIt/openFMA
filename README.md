# openFMA

开源多精度 Fused Multiply-Add（FMA）运算单元

## 概述

本项目实现了一个多精度 FMA 运算单元，支持 FP32、FP16、FP8（E4M3）和 INT8 四种数据格式。设计采用分阶段渐进实现策略，核心思想是**最大化运算器复用**——乘法器、移位器、CSA 压缩器和加法器跨精度共享。

## 支持的操作模式

| `mode[2:0]` | 操作模式 | 说明 |
|-------------|---------|------|
| `000` | FP32 FMA | `result = A×B + C`（单精度浮点） |
| `001` | FP16 FMA | `result = A×B + C`（半精度浮点） |
| `010` | FP8×4 FMA | 四路并行 FP8 FMA |
| `011` | INT8 MAD | `result = A_int×B_int + C_int`（8位整数乘加） |
| `100` | FP16→FP32 Acc | FP16×FP16 累加到 FP32 |
| `101` | FP8×4→FP32 Acc | 四路 FP8×FP8 累加到 FP32 |

## 目录结构

```
openFMA/
├── doc/
│   └── design_doc.md           # 设计文档
├── rtl/
│   ├── fma_top.v               # 顶层模块
│   ├── fma_fp32_stage1.v       # 流水线第一级：乘法
│   ├── fma_fp32_stage2.v       # 流水线第二级：对阶/CSA
│   ├── fma_fp32_stage3.v       # 流水线第三级：规格化/舍入
│   ├── fma_fp16_preprocessor.v # FP16 前端处理
│   ├── fma_fp8_preprocessor.v  # FP8 前端处理
│   ├── fma_int8_datapath.v     # INT8 数据通路
│   ├── fma_accumulator.v       # 混合精度累加器
│   ├── multiplier_48bit.v      # 48bit 共享乘法器
│   ├── shifter_50bit.v         # 50bit 移位器
│   ├── csa_3_2.v               # 3-2 CSA 压缩器
│   ├── adder_50bit.v           # 50bit 加法器
│   ├── normalizer_lzd.v        # 规格化器/前导零检测
│   └── special_value_handler.v # 特殊值处理
├── tb/
│   └── tb_fma_top.v            # 测试平台
└── or.md                       # 原始需求文档
```

## 仿真运行

使用 Icarus Verilog 编译和仿真：

```bash
# 编译
iverilog -o fma_sim -g2012 rtl/*.v tb/tb_fma_top.v

# 运行仿真
vvp fma_sim
```

## 设计特点

- **三级流水线**：乘法 → 对阶/CSA → 规格化/舍入，每周期产生一个结果
- **运算器复用**：48bit 乘法器、50bit 移位器、CSA 压缩器、50bit 加法器跨精度共享
- **IEEE 754 兼容**：支持 NaN、Inf、零、denormal 等特殊值处理
- **Round to Nearest Even**：采用 IEEE 754 默认舍入模式
