/**
 * FMA (Fused Multiply-Add) C 语言参考模型
 * ==========================================
 *
 * 这是 openFMA 硬件设计的"周期精确"软件模型。
 * 和 Python 版本不同，C 版本模拟了硬件的 3 级流水线行为。
 *
 * 写给大一同学: C 语言的位操作 (&, |, ^, <<, >>) 和硬件中的逻辑门
 * 是一一对应的，因此 C 模型比 Python 模型更"像"硬件。
 *
 * 编译: gcc -o fma_model fma_model.c -lm
 * 运行: ./fma_model
 */

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

// ============================================================
// 第1部分：数据类型定义（模拟硬件中的信号）
// ============================================================

/**
 * FP32 解包后的结构体
 *
 * 对应 Verilog 中的 wire [31:0] A 被拆成 wire A_sign, wire [7:0] A_exp, ...
 */
typedef struct {
    uint8_t sign;      // 1-bit 符号
    int16_t exp;       // 指数（有符号，真实指数值）
    uint32_t mant;     // 尾数（23-bit, 不含隐含的1）
} FP32Fields;

/**
 * FMA 流水线结构体
 *
 * 硬件的 3 级流水线：Stage1 → Stage2 → Stage3
 * 每个 stage 锁存计算结果，在下一个时钟周期传给下一级。
 */
typedef struct {
    // ---- Stage 1 寄存器 ----
    uint8_t  s1_valid;
    uint32_t s1_result_sign;
    int32_t  s1_exp_p;          // 乘积指数
    int32_t  s1_exp_c;          // C 的指数
    uint64_t s1_product_48;     // 24x24=48-bit 乘积
    uint32_t s1_mant_c;         // C 的尾数
    uint8_t  s1_sign_c;         // C 的符号

    // 特殊值标记
    uint8_t s1_nan;
    uint8_t s1_inf;
    uint8_t s1_zero;

    // ---- Stage 2 寄存器 ----
    uint8_t  s2_valid;
    uint32_t s2_result_sign;
    int32_t  s2_result_exp;
    uint64_t s2_result_mant;    // 50-bit 结果尾数

    // ---- Stage 3 寄存器 ----
    uint8_t  s3_valid;
    uint32_t s3_result;         // 最终 32-bit FP32 结果
} FMA_Pipeline;


// ============================================================
// 第2部分：解包/打包函数
// ============================================================

/**
 * 把 IEEE 754 FP32 bit 模式拆解成 sign/exp/mant 三个字段
 *
 * 例如输入 0x3F800000:
 *   sign = 0 (bit31)
 *   exp  = 127 (bit30-23 = 01111111)
 *   mant = 0   (bit22-0)
 *
 * 这就是 1.0 在计算机里的表示！
 */
FP32Fields fp32_unpack(uint32_t x) {
    FP32Fields f;
    f.sign = (x >> 31) & 1;
    f.exp  = (int16_t)((x >> 23) & 0xFF);
    f.mant = x & 0x7FFFFF;
    return f;
}

/** 打包: sign/exp/mant → 32-bit IEEE 754 */
uint32_t fp32_pack(uint8_t sign, uint8_t exp, uint32_t mant) {
    return ((uint32_t)sign << 31) | ((uint32_t)(exp & 0xFF) << 23) | (mant & 0x7FFFFF);
}

/** FP16 解包 */
void fp16_unpack(uint16_t x, uint8_t *sign, int16_t *exp, uint16_t *mant) {
    *sign = (x >> 15) & 1;
    *exp  = (int16_t)((x >> 10) & 0x1F);
    *mant = x & 0x3FF;
}

/** FP16 打包 */
uint16_t fp16_pack(uint8_t sign, uint8_t exp, uint16_t mant) {
    return ((uint16_t)sign << 15) | ((uint16_t)(exp & 0x1F) << 10) | (mant & 0x3FF);
}

/** FP8 E4M3 解包 */
void fp8_unpack(uint8_t x, uint8_t *sign, int16_t *exp, uint8_t *mant) {
    *sign = (x >> 7) & 1;
    *exp  = (int16_t)((x >> 3) & 0xF);
    *mant = x & 0x7;
}


// ============================================================
// 第3部分：特殊值检测
// ============================================================

int fp32_is_nan(uint32_t x)     { FP32Fields f = fp32_unpack(x); return f.exp == 255 && f.mant != 0; }
int fp32_is_inf(uint32_t x)     { FP32Fields f = fp32_unpack(x); return f.exp == 255 && f.mant == 0; }
int fp32_is_zero(uint32_t x)    { FP32Fields f = fp32_unpack(x); return f.exp == 0 && f.mant == 0; }

int fp16_is_nan(uint16_t x)     { uint8_t s; int16_t e; uint16_t m; fp16_unpack(x, &s, &e, &m); return e == 31 && m != 0; }
int fp16_is_inf(uint16_t x)     { uint8_t s; int16_t e; uint16_t m; fp16_unpack(x, &s, &e, &m); return e == 31 && m == 0; }


// ============================================================
// 第4部分：舍入函数 RNE (Round to Nearest Even)
// ============================================================

/**
 * 就近舍入到偶数
 *
 * 参数:
 *   value     - 待舍入的大整数（低位是小数部分）
 *   frac_bits - 小数部分占多少位
 * 返回: 整数部分（小数部分已舍去）
 *
 * 算法和 Python 版本完全一样，但用 C 的移位操作实现。
 */
uint64_t rne(uint64_t value, int frac_bits) {
    if (frac_bits == 0) return value;

    uint64_t integer_part = value >> frac_bits;
    uint64_t frac_part    = value & ((1ULL << frac_bits) - 1);
    uint64_t half         = 1ULL << (frac_bits - 1);

    if (frac_part > half) {
        return integer_part + 1;
    } else if (frac_part < half) {
        return integer_part;
    } else {
        // 正好一半 → 看整数最低位
        if (integer_part & 1)
            return integer_part + 1;
        else
            return integer_part;
    }
}


// ============================================================
// 第5部分：FP32 FMA 核心实现（对应硬件 3 级流水线）
// ============================================================

/**
 * FP32 FMA 全流程计算
 *
 * 这个函数模拟 hardware pipeline:
 *   cycle 1: 乘法 + 特殊值检查 (Stage 1)
 *   cycle 2: 对齐 + 相加 (Stage 2)
 *   cycle 3: 规格化 + 舍入 (Stage 3)
 *   cycle 4: 结果输出
 *
 * 返回值: 32-bit IEEE 754 格式
 */
uint32_t fp32_fma_compute(uint32_t a, uint32_t b, uint32_t c) {
    FP32Fields fa = fp32_unpack(a);
    FP32Fields fb = fp32_unpack(b);
    FP32Fields fc = fp32_unpack(c);

    int a_nan = fp32_is_nan(a), b_nan = fp32_is_nan(b), c_nan = fp32_is_nan(c);
    int a_inf = fp32_is_inf(a), b_inf = fp32_is_inf(b), c_inf = fp32_is_inf(c);
    int a_zero = fp32_is_zero(a), b_zero = fp32_is_zero(b);

    // ------- 特殊值处理 -------
    if (a_nan || b_nan || c_nan)
        return 0x7FC00000;  // NaN → 标准静默 NaN

    if ((a_zero && b_inf) || (a_inf && b_zero))
        return 0x7FC00000;  // 0×∞ → NaN

    if (a_inf || b_inf) {
        uint8_t sign_ab = fa.sign ^ fb.sign;
        if (c_inf && (sign_ab != fc.sign))
            return 0x7FC00000;  // ∞ + (-∞) → NaN
        return (sign_ab << 31) | (255 << 23);  // 无穷大
    }

    if (c_inf) return c;

    // ------- Stage 1: 尾数乘法 -------
    uint8_t  sign_p = fa.sign ^ fb.sign;

    // 展开尾数为 24-bit（加上隐含的1）
    uint32_t mant_a = (fa.exp == 0) ? fa.mant : (fa.mant | (1 << 23));
    uint32_t mant_b = (fb.exp == 0) ? fb.mant : (fb.mant | (1 << 23));

    // 真实指数值
    int32_t exp_a = (fa.exp == 0) ? -126 : (fa.exp - 127);
    int32_t exp_b = (fb.exp == 0) ? -126 : (fb.exp - 127);
    int32_t exp_p = exp_a + exp_b;

    // 24-bit × 24-bit = 48-bit —— 这是 FMA 的关键：全精度乘积！
    uint64_t product_48 = (uint64_t)mant_a * mant_b;

    // ------- Stage 2: 对齐 + 相加 -------
    uint32_t mant_c = (fc.exp == 0) ? fc.mant : (fc.mant | (1 << 23));
    int32_t  exp_c  = (fc.exp == 0) ? -126 : (fc.exp - 127);

    // 乘积扩展到 50-bit (binary point at bit 48):
    //   product_value = product_50 / 2^48 * 2^exp_p
    uint64_t product_50 = product_48 << 2;
    int32_t result_exp = exp_p;

    // C 对齐到和乘积相同的 binary point (bit 48):
    //   C_value = mant_c / 2^23 * 2^exp_c
    //   c_base = mant_c << 25 (即 mant_c / 2^23 = c_base / 2^48)
    uint64_t c_base_50 = (uint64_t)mant_c << 25;

    uint64_t result_50;
    uint8_t  result_sign;

    if (exp_p >= exp_c) {
        int32_t shift_amt = exp_p - exp_c;
        uint64_t c_aligned = (shift_amt >= 50) ? 0 : (c_base_50 >> shift_amt);
        if (sign_p == fc.sign) {
            result_50 = product_50 + c_aligned;
            result_sign = sign_p;
        } else {
            if (product_50 >= c_aligned) {
                result_50 = product_50 - c_aligned;
                result_sign = sign_p;
            } else {
                result_50 = c_aligned - product_50;
                result_sign = fc.sign;
            }
        }
    } else {
        int32_t shift_amt = exp_c - exp_p;
        uint64_t p_aligned = (shift_amt >= 50) ? 0 : (product_50 >> shift_amt);
        result_exp = exp_c;
        if (sign_p == fc.sign) {
            result_50 = c_base_50 + p_aligned;
            result_sign = fc.sign;
        } else {
            if (c_base_50 >= p_aligned) {
                result_50 = c_base_50 - p_aligned;
                result_sign = fc.sign;
            } else {
                result_50 = p_aligned - c_base_50;
                result_sign = sign_p;
            }
        }
    }

    // ------- Stage 3: 规格化 + 舍入 -------
    if (result_50 == 0)
        return (uint32_t)result_sign << 31;

    // 找最高位1 (binary point at bit 48)
    int lead_pos = -1;
    for (int i = 49; i >= 0; i--) {
        if ((result_50 >> i) & 1) { lead_pos = i; break; }
    }
    int shift = lead_pos - 48;

    if (shift > 0)      { result_50 >>= shift; result_exp += shift; }
    else if (shift < 0) { result_50 <<= -shift; result_exp -= -shift; }

    // mantissa_val = rne(result_50, 25) — 舍去低25位, 保留高25位(1+23+1 guard)
    uint64_t mantissa_val = rne(result_50, 25);

    // 舍入溢出检查: bit24 置位说明值>=2, 需要右移并调整指数
    if (mantissa_val >> 24) {
        mantissa_val >>= 1;
        result_exp += 1;
    }

    int32_t final_exp = result_exp + 127;
    uint32_t final_mant = (uint32_t)(mantissa_val & 0x7FFFFF);

    if (final_exp >= 255)
        return ((uint32_t)result_sign << 31) | (255U << 23);
    if (final_exp <= 0)
        return (uint32_t)result_sign << 31;

    return ((uint32_t)result_sign << 31) | ((uint32_t)final_exp << 23) | final_mant;
}


// ============================================================
// 第6部分：FP16 FMA（用 float 快速计算）
// ============================================================

float fp16_to_float(uint16_t x) {
    uint8_t s; int16_t e; uint16_t m;
    fp16_unpack(x, &s, &e, &m);
    if (e == 0 && m == 0) return s ? -0.0f : 0.0f;
    if (e == 0)  return (s ? -1.0f : 1.0f) * (float)(m) / 1024.0f * powf(2.0f, -14.0f);
    if (e == 31) return m ? NAN : (s ? -INFINITY : INFINITY);
    return (s ? -1.0f : 1.0f) * (1.0f + (float)m / 1024.0f) * powf(2.0f, (float)(e - 15));
}

uint16_t float_to_fp16(float f) {
    if (f == 0.0f) return 0;
    uint8_t s = (f < 0) ? 1 : 0;
    if (s) f = -f;

    int e_raw = (int)floorf(log2f(f));
    int exp = e_raw + 15;

    if (exp >= 31) return (s << 15) | (31 << 10);
    if (exp <= 0)  return (s << 15);

    int mantissa = (int)((f / powf(2.0f, (float)e_raw) - 1.0f) * 1024.0f + 0.5f);
    if (mantissa >= 1024) { exp++; mantissa = 0; }

    return (s << 15) | (exp << 10) | mantissa;
}

uint16_t fp16_fma_compute(uint16_t a, uint16_t b, uint16_t c) {
    float result = fp16_to_float(a) * fp16_to_float(b) + fp16_to_float(c);
    return float_to_fp16(result);
}


// ============================================================
// 第7部分：INT8 MAD（深度学习训练的基本运算！）
// ============================================================

/**
 * INT8 MAD: 有符号 8-bit 乘加
 * 这就是 AI 芯片（GPU/NPU）做矩阵乘法的基本操作！
 */
int16_t int8_mad_compute(int8_t a, int8_t b, int16_t c) {
    return (int16_t)a * (int16_t)b + c;
}


// ============================================================
// 第8部分：自测试
// ============================================================

int main() {
    printf("==================================================\n");
    printf("FMA C Model Self-Test\n");
    printf("==================================================\n");

    int passed = 0, total = 0;

    // Test 1: FP32 1.0 * 1.0 + 0.0 = 1.0
    total++;
    {
        uint32_t r = fp32_fma_compute(0x3F800000, 0x3F800000, 0x00000000);
        if (r == 0x3F800000) {
            printf("  PASS: 1.0 * 1.0 + 0.0 = 0x%08X\n", r);
            passed++;
        } else {
            printf("  FAIL: 1.0 * 1.0 + 0.0: got 0x%08X, expected 0x3F800000\n", r);
        }
    }

    // Test 2: FP32 2.0 * 3.0 + 4.0 = 10.0
    total++;
    {
        uint32_t a = 0x40000000;  // 2.0
        uint32_t b = 0x40400000;  // 3.0
        uint32_t c = 0x40800000;  // 4.0
        uint32_t r = fp32_fma_compute(a, b, c);
        // 10.0 = 0x41200000
        if (r == 0x41200000) {
            printf("  PASS: 2.0 * 3.0 + 4.0 = 10.0 (0x%08X)\n", r);
            passed++;
        } else {
            printf("  FAIL: 2.0 * 3.0 + 4.0: got 0x%08X\n", r);
        }
    }

    // Test 3: NaN propagation
    total++;
    {
        uint32_t nan = 0x7FC00000;
        uint32_t r = fp32_fma_compute(nan, 0x3F800000, 0x00000000);
        if (fp32_is_nan(r)) {
            printf("  PASS: NaN * 1.0 + 0.0 = NaN\n");
            passed++;
        } else {
            printf("  FAIL: NaN propagation failed\n");
        }
    }

    // Test 4: 0 * Inf = NaN
    total++;
    {
        uint32_t r = fp32_fma_compute(0x00000000, 0x7F800000, 0x00000000);
        if (fp32_is_nan(r)) {
            printf("  PASS: 0 * Inf + 0 = NaN\n");
            passed++;
        } else {
            printf("  FAIL: 0 * Inf should be NaN\n");
        }
    }

    // Test 5: FP16 1.0 * 1.0 + 0.0 = 1.0
    total++;
    {
        uint16_t r = fp16_fma_compute(0x3C00, 0x3C00, 0x0000);
        if (r == 0x3C00) {
            printf("  PASS: FP16 1.0 * 1.0 + 0.0 = 0x%04X\n", r);
            passed++;
        } else {
            printf("  FAIL: FP16 1.0 * 1.0 + 0.0: got 0x%04X\n", r);
        }
    }

    // Test 6: INT8 2*3+4=10
    total++;
    {
        int16_t r = int8_mad_compute(2, 3, 4);
        if (r == 10) {
            printf("  PASS: INT8 2*3+4 = %d\n", r);
            passed++;
        } else {
            printf("  FAIL: INT8 2*3+4: got %d\n", r);
        }
    }

    printf("\n%d/%d tests passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}
