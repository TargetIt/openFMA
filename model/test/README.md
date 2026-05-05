# FMA 测试环境指南

> 芯片验证的"三步法"：Python 出题 → C 做题 → Verilog 对答案

## 文件说明

| 文件 | 干什么 |
|------|--------|
| `gen_vectors.py` | 生成测试向量（输入+标准答案） |
| `compare.py` | 比较 Python/C/RTL 三方结果是否一致 |
| `vectors/` | 存储生成的测试向量 |

## 快速开始

### Step 1：生成测试向量

```bash
cd model/test
python3 gen_vectors.py
```

生成 180 个测试向量：
- `vectors/fp32_vectors.txt` — 100 个 FP32 测试
- `vectors/fp16_vectors.txt` — 50 个 FP16 测试
- `vectors/int8_vectors.txt` — 30 个 INT8 测试

### Step 2：运行三方验证

```bash
python3 compare.py
```

输出示例：
```
============================================================
FMA Cross-Model Verification
============================================================

[1/2] Running Python model...
  Python: 5 passed, 0 failed

[2/2] Running C model...
  C:      6 passed, 0 failed

============================================================
Verification Summary
============================================================
Model        Passed     Failed
--------------------------------
Python       5          0         ✅
C            6          0         ✅

Consistency:
  ✅ Python and C results match!
```

### Step 3：与 RTL 仿真对比

1. 先跑 Verilog 仿真生成 `wave.vcd`
2. 用脚本提取 RTL 输出
3. 和 Python 的标准答案对比

```bash
# 如果 RTL 仿真日志中有结果输出:
grep "result =" phase2_sim/test_results.log | python3 compare.py --from-rtl
```

## 测试向量格式

### FP32 (fp32_vectors.txt)
```
A_hex B_hex C_hex expected_hex
3F800000 3F800000 00000000 3F800000
```
含义: `1.0 * 1.0 + 0.0 = 1.0`

### FP16 (fp16_vectors.txt)
```
A_hex B_hex C_hex expected_hex
3C00 3C00 0000 3C00
```
含义: `FP16(1.0) * FP16(1.0) + FP16(0.0) = FP16(1.0)`

### INT8 (int8_vectors.txt)
```
a_int8 b_int8 c_int16 expected_int16
2 3 4 10
```
含义: `2 * 3 + 4 = 10`

## 验证原理

```
               ┌─────────────┐
               │  Python 模型  │ ← 最可信（数学上正确）
               │  (标准答案)   │
               └──────┬──────┘
                      │ 比较
          ┌───────────┼───────────┐
          │           │           │
    ┌─────▼─────┐ ┌──▼────┐ ┌───▼──────┐
    │  C 模型    │ │ 一致？│ │ RTL 仿真 │
    │ (快速批量) │ │  ✅   │ │ (硬件实现)│
    └───────────┘ └───────┘ └──────────┘
```

如果 Python 和 C 一致 → 算法没问题
如果 Python 和 RTL 一致 → 硬件实现没问题
如果 C 和 RTL 一致 → 综合前后一致

这就是芯片行业的"黄金验证流程"。
