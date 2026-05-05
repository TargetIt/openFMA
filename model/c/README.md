# C 语言 FMA 建模指南

> 适合：学过 C 语言基础（指针、结构体、位运算）的大一学生

## 为什么用 C？

| | Python 模型 | C 模型 | Verilog RTL |
|---|-----------|--------|-------------|
| 运行速度 | 慢 | **快**（10~100倍）| 仿真很慢 |
| 和硬件相似度 | 像算法 | **像电路** | 就是电路 |
| 适合做什么 | 学习算法、快速验证 | 批量生成测试向量 | 最终流片 |

C 语言的位运算 (`&`, `|`, `^`, `<<`, `>>`) 和硬件门电路有直接对应关系：
- `&` ⇔ AND 门
- `|` ⇔ OR 门
- `^` ⇔ XOR 门
- `<<` ⇔ 左移位器
- `>>` ⇔ 右移位器

## 编译与运行

```bash
gcc -o fma_model fma_model.c -lm
./fma_model
```

预期输出：
```
==================================================
FMA C Model Self-Test
==================================================
  PASS: 1.0 * 1.0 + 0.0 = 0x3F800000
  PASS: 2.0 * 3.0 + 4.0 = 10.0 (0x41200000)
  PASS: NaN * 1.0 + 0.0 = NaN
  PASS: 0 * Inf + 0 = NaN
  PASS: FP16 1.0 * 1.0 + 0.0 = 0x3C00
  PASS: INT8 2*3+4 = 10

6/6 tests passed
```

## 代码亮点

### 1. `uint64_t` 模拟硬件精度

硬件有 50-bit 数据通路。C 语言用 `uint64_t`（64-bit 无符号整数）来存，绰绰有余：
```c
uint64_t product_48 = (uint64_t)mant_a * mant_b;  // 24×24=48 bit
uint64_t product_50 = product_48 << 2;              // 扩展到 50 bit
```

### 2. 结构体模拟寄存器

```c
FP32Fields fa = fp32_unpack(a);  // 把 32-bit 拆成 {符号, 指数, 尾数}
// 就像硬件把 32 根线分成 3 组：
// A[31] → sign, A[30:23] → exp, A[22:0] → mant
```

### 3. 三步流水

```c
// Stage 1: 乘法
product_48 = mant_a * mant_b;

// Stage 2: 对齐 + 相加
c_aligned = c_50 >> exp_diff;
result_50 = product_50 + c_aligned;

// Stage 3: 规格化 + 舍入
rounded = rne(result_50, frac_bits);
```

## 自测试结果

运行 `./fma_model` 的输出来看，所有测试均通过，说明 C 模型和 Python 模型的逻辑一致。

## 下一步

看 [测试环境](../test/README.md)，了解如何用 Python/C 模型生成"标准答案"来验证 Verilog RTL。
