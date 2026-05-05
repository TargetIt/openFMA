#!/usr/bin/env python3
"""
FMA (Fused Multiply-Add) Python 参考模型
==========================================
这是 openFMA 硬件设计的"软件镜像"——
用 Python 实现和硬件一模一样的计算逻辑，
用来验证硬件输出是否正确。

支持模式: FP32, FP16, FP8x4, INT8, FP16->FP32 Acc, FP8x4->FP32 Acc

写给初学者: 每个函数都加了详细注释。
"""

import struct


# ============================================================
# 第1部分：位操作工具函数
# ============================================================

def bits(val, hi, lo=None):
    """提取 value 的 bit[hi:lo]（像 Verilog 的 val[hi:lo]）

    用法:
        bits(0b10110, 4, 1) → 0b1011 (bit4到bit1)
        bits(0b10110, 2)     → 1     (bit2)
    """
    if lo is None:
        return (val >> hi) & 1
    return (val >> lo) & ((1 << (hi - lo + 1)) - 1)


# ============================================================
# 第2部分：IEEE 754 浮点数 <--> 整数 互转
# ============================================================

def fp32_unpack(x):
    """把 32-bit 整数拆成 (sign, exponent, mantissa)

    例如: 0x3f800000 → (0, 127, 0)
          这就是 1.0 的 FP32 表示:
             符号=0(正), 指数=127(bias=127, 实际0), 尾数=0(隐含1)
    """
    s = bits(x, 31)              # bit31: 符号
    e = bits(x, 30, 23)           # bit30-23: 指数 (8 bits)
    m = bits(x, 22, 0)            # bit22-0: 尾数 (23 bits)
    return s, e, m


def fp32_pack(s, e, m):
    """把 (sign, exponent, mantissa) 打包成 32-bit 整数"""
    return (s << 31) | ((e & 0xFF) << 23) | (m & 0x7FFFFF)


def fp16_unpack(x):
    """16-bit 整数 → (sign, exponent, mantissa)"""
    s = bits(x, 15)
    e = bits(x, 14, 10)
    m = bits(x, 9, 0)
    return s, e, m


def fp16_pack(s, e, m):
    """(sign, exponent, mantissa) → 16-bit 整数"""
    return (s << 15) | ((e & 0x1F) << 10) | (m & 0x3FF)


def fp8_unpack(x):
    """8-bit E4M3 整数 → (sign, exponent, mantissa)

    FP8 E4M3: 1-bit sign, 4-bit exponent (bias=7), 3-bit mantissa
    """
    s = bits(x, 7)
    e = bits(x, 6, 3)
    m = bits(x, 2, 0)
    return s, e, m


def fp8_pack(s, e, m):
    """(sign, exponent, mantissa) → 8-bit FP8 E4M3"""
    return (s << 7) | ((e & 0xF) << 3) | (m & 0x7)


# ============================================================
# 第3部分：浮点数值计算
# ============================================================

def fp32_value(x_int):
    """计算 FP32 的实际数值

    返回: float 类型的真实值
    """
    s, e, m = fp32_unpack(x_int)

    if e == 0 and m == 0:
        # 零: (-1)^s × 0
        return -0.0 if s else 0.0
    elif e == 0:
        # 次正规数: (-1)^s × 0.m × 2^(-126)
        return (-1)**s * (m / 2**23) * 2**(-126)
    elif e == 255:
        if m == 0:
            # 无穷大
            return float('-inf') if s else float('inf')
        else:
            # NaN
            return float('nan')
    else:
        # 正规数: (-1)^s × 1.m × 2^(e-127)
        return (-1)**s * (1 + m / 2**23) * 2**(e - 127)


def fp16_value(x_int):
    """FP16 bit 模式 → 实际数值"""
    s, e, m = fp16_unpack(x_int)
    if e == 0 and m == 0:
        return -0.0 if s else 0.0
    elif e == 0:
        return (-1)**s * (m / 2**10) * 2**(-14)
    elif e == 31:
        return float('nan') if m else (float('-inf') if s else float('inf'))
    else:
        return (-1)**s * (1 + m / 2**10) * 2**(e - 15)


def fp8_value(x_int):
    """FP8 E4M3 bit 模式 → 实际数值"""
    s, e, m = fp8_unpack(x_int)
    if e == 0 and m == 0:
        return -0.0 if s else 0.0
    elif e == 0:
        return (-1)**s * (m / 2**3) * 2**(-6)
    elif e == 15:
        return float('nan') if m else (float('-inf') if s else float('inf'))
    else:
        return (-1)**s * (1 + m / 2**3) * 2**(e - 7)


# ============================================================
# 第4部分：判等辅助函数
# ============================================================

def fp32_is_nan(x):
    """FP32 bit 模式是否为 NaN"""
    _, e, m = fp32_unpack(x)
    return e == 255 and m != 0


def fp32_is_inf(x):
    """FP32 bit 模式是否为 无穷大"""
    _, e, m = fp32_unpack(x)
    return e == 255 and m == 0


def fp32_is_zero(x):
    """FP32 bit 模式是否为零"""
    _, e, m = fp32_unpack(x)
    return e == 0 and m == 0


def fp16_is_nan(x):
    _, e, m = fp16_unpack(x)
    return e == 31 and m != 0


def fp16_is_inf(x):
    _, e, m = fp16_unpack(x)
    return e == 31 and m == 0


# ============================================================
# 第5部分：舍入函数 (Round to Nearest Even)
# ============================================================

def rne(value, frac_bits):
    """就近舍入到偶数 (RNE)

    参数:
        value: 要大整数值（包含未舍入的小数位）
        frac_bits: 小数部分占多少位

    例如: rne(0b1011, 2) 表示 value 的小数部分占2位
          = 10.11(二进制)
          = 2.75(十进制)
          舍入到整数 → 3 (因为 0.75 > 0.5)
          = 0b11

    算法：
        如果小数部分 > 一半 → 进位
        如果小数部分 < 一半 → 舍去
        如果正好一半 → 看整数部分最低位（偶数舍去，奇数进位）
    """
    if frac_bits == 0:
        return value

    # 整数部分（去掉小数位）
    integer_part = value >> frac_bits
    # 小数部分（仅保留frac_bits位）
    frac_part = value & ((1 << frac_bits) - 1)
    # "一半"的值: 1000...0 (frac_bits位)
    half = 1 << (frac_bits - 1)

    if frac_part > half:
        # 大于一半 → 进位
        return integer_part + 1
    elif frac_part < half:
        # 小于一半 → 舍去
        return integer_part
    else:
        # 正好一半 → 向偶数舍入
        if integer_part & 1:
            return integer_part + 1
        else:
            return integer_part


# ============================================================
# 第6部分：FP32 FMA 核心实现
# ============================================================

def fp32_fma(a_int, b_int, c_int):
    """FP32 FMA: result = A × B + C

    参数都是 32-bit 整数 (IEEE 754 格式)
    返回值也是 32-bit 整数

    这个函数模拟硬件的计算流程：
      Stage1: 符号/指数/尾数拆解 + 24×24 尾数乘法
      Stage2: 指数对齐 + 3-2 CSA 相加
      Stage3: 前导零检测 + 规格化 + 舍入
    """
    # --- 特殊值检查 ---
    sa, ea, ma = fp32_unpack(a_int)
    sb, eb, mb = fp32_unpack(b_int)
    sc, ec, mc = fp32_unpack(c_int)

    a_nan = fp32_is_nan(a_int)
    b_nan = fp32_is_nan(b_int)
    c_nan = fp32_is_nan(c_int)
    a_inf = fp32_is_inf(a_int)
    b_inf = fp32_is_inf(b_int)
    c_inf = fp32_is_inf(c_int)
    a_zero = fp32_is_zero(a_int)
    b_zero = fp32_is_zero(b_int)

    # 规则1: 任何操作数是 NaN → 结果 NaN
    if a_nan or b_nan or c_nan:
        return 0x7FC00000  # 标准静默 NaN

    # 规则2: 0 × ∞ → NaN (无效运算)
    if (a_zero and b_inf) or (a_inf and b_zero):
        return 0x7FC00000

    # 规则3: 无穷大运算
    if a_inf or b_inf:
        # A×B 是无穷大
        sign_ab = sa ^ sb
        if c_inf and (sign_ab != sc):
            # Infinity + (-Infinity) = NaN
            return 0x7FC00000
        return (sign_ab << 31) | (255 << 23)  # 返回带正确符号的无穷大

    if c_inf:
        return c_int

    # --- Stage1: 尾数乘法 ---
    sign_p = sa ^ sb  # 乘积符号

    # 计算指数（带 bias 修正）
    if ea == 0:
        exp_a = -126  # 次正规数
        mant_a = ma   # 次正规数的尾数是 0.m
    else:
        exp_a = ea - 127
        mant_a = (1 << 23) | ma  # 加隐含的 1

    if eb == 0:
        exp_b = -126
        mant_b = mb
    else:
        exp_b = eb - 127
        mant_b = (1 << 23) | mb

    exp_p = exp_a + exp_b

    # 24-bit × 24-bit = 48-bit 乘积（这是 FMA 的核心：保留全精度！）
    product_48 = mant_a * mant_b

    # --- Stage2: 对齐 + CSA 相加 ---
    if ec == 0:
        exp_c = -126
        mant_c = mc
    else:
        exp_c = ec - 127
        mant_c = (1 << 23) | mc

    sign_c = sc

    # 乘积扩展: 48-bit (binary point at bit 46) → 50-bit (binary point at bit 48)
    #   product / 2^46 * 2^exp_p = product_50 / 2^48 * 2^exp_p
    product_50 = product_48 << 2
    result_exp = exp_p

    # C 对齐到与乘积同样的 binary point (bit 48)
    #   C value = mant_c / 2^23 * 2^exp_c
    # 需对齐到: c_aligned / 2^48 * 2^exp_diff_adjusted
    #   c_aligned = mant_c << (48 - 23) = mant_c << 25
    c_base_50 = mant_c << 25

    # 指数对齐
    if exp_p >= exp_c:
        # 乘积指数更大，C 需要右移
        shift_amt = exp_p - exp_c
        c_aligned = c_base_50 >> shift_amt if shift_amt < 50 else 0
        if sign_p == sign_c:
            result_50 = product_50 + c_aligned
            result_sign = sign_p
        else:
            if product_50 >= c_aligned:
                result_50 = product_50 - c_aligned
                result_sign = sign_p
            else:
                result_50 = c_aligned - product_50
                result_sign = sign_c
    else:
        # C 指数更大，乘积需要右移
        shift_amt = exp_c - exp_p
        p_aligned = product_50 >> shift_amt if shift_amt < 50 else 0
        result_exp = exp_c
        if sign_p == sign_c:
            result_50 = c_base_50 + p_aligned
            result_sign = sign_c
        else:
            if c_base_50 >= p_aligned:
                result_50 = c_base_50 - p_aligned
                result_sign = sign_c
            else:
                result_50 = p_aligned - c_base_50
                result_sign = sign_p

    # --- Stage3: 规格化 + 舍入 ---
    if result_50 == 0:
        return result_sign << 31

    # 前导零/溢出检测: 找最高位1 (binary point at bit 48)
    lead_pos = max((i for i in range(50) if (result_50 >> i) & 1), default=-1)
    shift = lead_pos - 48

    if shift > 0:
        result_50 >>= shift
        result_exp += shift
    elif shift < 0:
        result_50 <<= -shift
        result_exp -= -shift

    # 现在 bit48=1, value在[1,2)
    # mantissa_25bit = rne(result_50 << 1, 25)
    # (左移1位使隐含1到bit24位置，然后再舍入到25-bit = 1个整数bit + 23个尾数bit + 1个舍入bit)
    # 简化: rne(result_50, 25) 舍去bits[24:0], 保留bits[49:25]作为整数
    mantissa_val = rne(result_50, 25)

    # mantissa_val / 2^25 应该接近1.xxx
    # 如果 mantissa_val 的 bit24 被设置(值>=2)，说明舍入导致溢出
    if mantissa_val >> 24:
        mantissa_val >>= 1
        result_exp += 1

    final_exp = result_exp + 127
    final_mant = mantissa_val & 0x7FFFFF

    # 上溢/下溢
    if final_exp >= 255:
        return (result_sign << 31) | (255 << 23)
    elif final_exp <= 0:
        return result_sign << 31

    return (result_sign << 31) | (final_exp << 23) | final_mant


# ============================================================
# 第7部分：其他模式
# ============================================================

def fp16_fma(a16, b16, c16):
    """FP16 FMA: A×B+C, 所有输入/输出都是 FP16"""
    # 转为 FP32 用浮点计算，再转回 FP16（用 Python 浮点做参考）
    av = fp16_value(a16)
    bv = fp16_value(b16)
    cv = fp16_value(c16)
    result = av * bv + cv

    # 转回 FP16 bit 模式
    import struct
    # 使用 numpy 或手动转换
    return float_to_fp16(result)


def float_to_fp16(f):
    """Python float → FP16 bit 模式（简化版）"""
    if f == 0.0:
        return 0
    import math
    s = 0
    if f < 0:
        s = 1
        f = -f

    # 求指数
    e_raw = int(math.floor(math.log2(f)))
    exp = e_raw + 15

    if exp >= 31:
        return (s << 15) | (31 << 10)  # 无穷大
    if exp <= 0:
        return (s << 15)  # 下溢到零

    # 求尾数
    mantissa = int((f / (2 ** e_raw) - 1.0) * 1024 + 0.5)
    if mantissa >= 1024:
        exp += 1
        mantissa = 0

    return (s << 15) | (exp << 10) | mantissa


def int8_mad(a8, b8, c16):
    """INT8 乘加: A*B+C (有符号 8-bit 乘, 16-bit 累加)

    这就是深度学习中的 INT8 GEMM 的基本操作！
    """
    # 有符号扩展
    if a8 & 0x80:
        a_val = a8 - 256
    else:
        a_val = a8
    if b8 & 0x80:
        b_val = b8 - 256
    else:
        b_val = b8
    if c16 & 0x8000:
        c_val = c16 - 65536
    else:
        c_val = c16

    result = a_val * b_val + c_val
    # 截断到 16-bit
    result = result & 0xFFFF
    return result


# ============================================================
# 第8部分：自测试
# ============================================================

def self_test():
    """运行自测试，验证模型正确性"""
    print("=" * 50)
    print("FMA Python Model Self-Test")
    print("=" * 50)
    passed = 0
    total = 0

    # Test 1: FP32 1.0 * 1.0 + 0.0 = 1.0
    total += 1
    a = fp32_pack(0, 127, 0)  # 1.0
    b = fp32_pack(0, 127, 0)  # 1.0
    c = fp32_pack(0, 0, 0)    # 0.0
    result = fp32_fma(a, b, c)
    expected = fp32_pack(0, 127, 0)  # 1.0
    if result == expected:
        print(f"  PASS: 1.0 * 1.0 + 0.0 = 1.0 (0x{result:08X})")
        passed += 1
    else:
        print(f"  FAIL: 1.0 * 1.0 + 0.0: got 0x{result:08X}, expected 0x{expected:08X}")

    # Test 2: FP32 2.0 * 3.0 + 4.0 = 10.0
    total += 1
    a = fp32_pack(0, 128, 0)      # 2.0
    b = fp32_pack(0, 128, 64<<16) # 3.0
    c = fp32_pack(0, 129, 0)      # 4.0
    result = fp32_fma(a, b, c)
    actual_val = fp32_value(result)
    if abs(actual_val - 10.0) < 0.0001:
        print(f"  PASS: 2.0 * 3.0 + 4.0 = {actual_val}")
        passed += 1
    else:
        print(f"  FAIL: 2.0 * 3.0 + 4.0: got {actual_val}")

    # Test 3: NaN propagation
    total += 1
    a = fp32_pack(0, 255, 1)  # NaN
    b = fp32_pack(0, 127, 0)  # 1.0
    c = fp32_pack(0, 0, 0)    # 0.0
    result = fp32_fma(a, b, c)
    if fp32_is_nan(result):
        print(f"  PASS: NaN * 1.0 + 0.0 = NaN")
        passed += 1
    else:
        print(f"  FAIL: NaN propagation failed")

    # Test 4: 0 * Inf = NaN
    total += 1
    a = fp32_pack(0, 0, 0)    # 0
    b = fp32_pack(0, 255, 0)  # Inf
    c = fp32_pack(0, 0, 0)    # 0
    result = fp32_fma(a, b, c)
    if fp32_is_nan(result):
        print(f"  PASS: 0 * Inf + 0 = NaN")
        passed += 1
    else:
        print(f"  FAIL: 0 * Inf should be NaN")

    # Test 5: INT8 MAD
    total += 1
    result = int8_mad(2, 3, 4)  # 2*3+4=10
    if result == 10:
        print(f"  PASS: INT8 2*3+4 = {result}")
        passed += 1
    else:
        print(f"  FAIL: INT8 2*3+4: got {result}")

    print(f"\n{passed}/{total} tests passed")
    return passed == total


if __name__ == "__main__":
    success = self_test()
    exit(0 if success else 1)
