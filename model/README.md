# openFMA 建模与验证平台

> **写给大一同学**: 这个目录是一套"从数学到代码到芯片"的完整学习材料。
> 不需要硬件背景 —— 只要会 Python 或 C 语言，就能理解芯片是怎么设计的。

## 学习路线图

```
第1步: 数学建模          第2步: 软件建模          第3步: 硬件验证
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ doc/          │      │ python/       │      │ test/         │
│ math_model.md │ ───→ │ fma_model.py  │ ───→ │ gen_vectors.py│
│              │      │ README.md     │      │ compare.py    │
│ 理解原理      │      │ 动手写代码     │      │ 验证正确性     │
└──────────────┘      └──────────────┘      └──────────────┘
                             │
                             ▼
                      ┌──────────────┐
                      │ c/            │
                      │ fma_model.c   │
                      │ README.md     │
                      │              │
                      │ 高性能批量生成 │
                      └──────────────┘
```

## 目录结构

```
model/
├── README.md                    ← 本文件（总览和学习路线）
├── doc/
│   └── math_model.md            ← 数学建模（含二进制/浮点数/舍入算法推导）
├── python/
│   ├── README.md                ← Python 建模入门指南
│   └── fma_model.py             ← Python bit-accurate 参考模型
├── c/
│   ├── README.md                ← C 建模入门指南
│   └── fma_model.c              ← C 语言周期精确参考模型
└── test/
    ├── README.md                ← 测试环境使用指南
    ├── gen_vectors.py           ← 测试向量生成器
    ├── compare.py               ← 三方一致性比较器
    └── vectors/                 ← 生成的测试向量
```

## 各建模的对比

| 维度 | 数学建模 | Python 建模 | C 建模 | Verilog RTL |
|------|---------|------------|--------|-------------|
| **写给人还是机器** | 人 | 人+机器 | 机器 | 机器 |
| **和硬件相似度** | 0% | 30% | 70% | 100% |
| **运行速度** | N/A | 慢 | 快 | 仿真很慢/芯片极快 |
| **主要用途** | 理解原理 | 快速原型验证 | 批量测试向量 | 物理实现 |
| **需要的知识** | 高数+进制 | Python基础 | C语言+位运算 | Verilog+电路 |
| **教学阶段** | 第1周 | 第2周 | 第3周 | 第4周 |

## 快速验证

```bash
# 1. Python 自测试
cd model/python && python3 fma_model.py

# 2. C 编译与测试
cd model/c && gcc -o fma_model fma_model.c -lm && ./fma_model

# 3. 生成测试向量
cd model/test && python3 gen_vectors.py

# 4. 三方一致性验证
cd model/test && python3 compare.py
```

## 给老师的教学建议

### 第一周：数学建模 (2课时)
- 第1课时: 二进制、科学计数法、IEEE 754 浮点格式
- 第2课时: FMA 算法推导（乘法→对齐→加法→规格化→舍入）
- 作业: 手算 `1.25 × 2.5 + 0.75` 的 FP32 表示

### 第二周：Python 建模 (2课时)
- 第1课时: 对照数学公式写 Python 代码
- 第2课时: 调试、验证、生成测试向量
- 作业: 给 `fp32_fma` 加 5 个新测试用例

### 第三周：C 建模 (2课时)
- 第1课时: C 语言位运算、结构体、流水线概念
- 第2课时: 批量生成 1000+ 测试向量
- 作业: 测量 Python vs C 的速度差异

### 第四周：RTL 对接 (2课时)
- 第1课时: 用 Python/C 测试向量验证 Verilog RTL
- 第2课时: 综合、PnR、物理验证
- 期末: 完成 RTL→GDS 全流程

## 相关资源

- [数学建模详细文档](doc/math_model.md)
- [Python 建模入门](python/README.md)
- [C 语言建模入门](c/README.md)
- [测试环境使用指南](test/README.md)
- [Verilog RTL 源码](../rtl/stage4/fma_top_stage4.v)
- [全流程运行脚本](../run_all.sh)
