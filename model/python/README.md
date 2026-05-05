# Python FMA 建模指南

> 适合：会 Python 基础语法的大一学生

## 这是什么？

`fma_model.py` 是 openFMA 硬件设计的**软件参考模型**。它用 Python 代码一字一句地"翻译"了硬件的工作方式。

**为什么需要它？**
- 硬件（Verilog）跑得慢，Python 跑得快 → 用来生成测试数据
- 硬件容易出错，Python 逻辑清晰 → 用来验证硬件是否正确
- 算法看得见摸得着 → 用来学 FMA 是怎么算的

## 快速开始

```bash
# 运行自测试
python3 fma_model.py
```

预期输出：
```
==================================================
FMA Python Model Self-Test
==================================================
  PASS: 1.0 * 1.0 + 0.0 = 1.0 (0x3F800000)
  PASS: 2.0 * 3.0 + 4.0 = 10.0
  PASS: NaN * 1.0 + 0.0 = NaN
  PASS: 0 * Inf + 0 = NaN
  PASS: INT8 2*3+4 = 10

5/5 tests passed
```

## 代码导航

| 函数 | 干什么的 | 行数 |
|------|---------|------|
| `bits(val, hi, lo)` | 提取二进制位 | ~5 |
| `fp32_unpack/pack` | 拆解/打包 FP32 | ~10 |
| `fp32_value` | FP32 bit → 实际数值 | ~20 |
| `fp32_fma` | **核心！** FP32 乘法+加法 | ~100 |
| `fp32_is_nan/inf/zero` | 判等工具 | ~10 |
| `rne` | 舍入函数 | ~20 |
| `int8_mad` | INT8 乘加 | ~10 |
| `self_test` | 自测试 | ~30 |

## 动手实验

### 实验1：手算一个 FMA

用 Python 交互模式试试：
```python
>>> from fma_model import *
>>> a = fp32_pack(0, 127, 0)   # 1.0
>>> b = fp32_pack(0, 128, 0)   # 2.0
>>> c = fp32_pack(0, 0, 0)     # 0.0
>>> result = fp32_fma(a, b, c)
>>> print(hex(result))          # 0x40000000 = 2.0
>>> print(fp32_value(result))   # 2.0
```

### 实验2：理解舍入误差

```python
>>> # 一个很小的数 + 一个很大的数
>>> a = fp32_pack(0, 100, 0)   # ~2^27
>>> b = fp32_pack(0, 100, 0)   # ~2^27
>>> c = fp32_pack(0, 50, 0)    # ~2^23 (小了16倍!)
>>> result = fp32_fma(a, b, c)
>>> # 精确值 = 2^54 + 2^23，但 FP32 只有 23-bit 尾数
>>> print(f"FMA result: {fp32_value(result):.1f}")
```

### 实验3：观察 NaN 传播

```python
>>> nan = fp32_pack(0, 255, 1)  # 随便造一个 NaN
>>> any_num = fp32_pack(0, 127, 0)  # 1.0
>>> result = fp32_fma(nan, any_num, any_num)
>>> print(fp32_is_nan(result))  # True — NaN 传染一切！
```

## 和硬件的关系

这个 Python 函数：
```python
def fp32_fma(a_int, b_int, c_int):
    # ... 100 行代码
    return result
```

对应 Verilog 中 535 行的 `fma_top_stage4` 模块。两者计算逻辑一致，但：
- Python 是一次函数调用；硬件是 3 级流水线
- Python 用大整数模拟 50-bit；硬件真有 50 根线
- Python 的 `if/else`；硬件是 `mux` + `case`

## 下一步

看完这个后，看看 [C 语言建模](../c/README.md)，它和硬件更接近（有真正的流水线周期模拟）。
