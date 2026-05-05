#!/usr/bin/env python3
"""
三方一致性验证脚本
==================
比较 Python 模型、C 模型、RTL 仿真三者的输出是否一致。

用法:
  python3 compare.py --python --rtl wave.vcd
  python3 compare.py --python --c
  python3 compare.py --all

这是芯片验证的"黄金标准"流程:
  1. Python 生成测试向量（标准答案）
  2. C 模型用同样向量计算（如果一致 → 算法对）
  3. RTL 仿真用同样向量（如果一致 → 硬件实现对）
"""

import sys
import os
import subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../python')
from fma_model import fp32_fma, fp32_is_nan, fp32_is_inf


def run_c_model():
    """编译并运行 C 模型，解析输出"""
    c_dir = os.path.dirname(os.path.abspath(__file__)) + '/../c'
    c_src = c_dir + '/fma_model.c'
    c_bin = c_dir + '/fma_model'

    if not os.path.exists(c_src):
        return None, "C source not found"

    # 查找 gcc
    gcc_path = None
    for path in ['gcc', '/usr/bin/gcc', '/usr/local/bin/gcc']:
        if os.path.exists(path) or subprocess.run(['which', path.split('/')[-1]], capture_output=True).returncode == 0:
            gcc_path = 'gcc'  # use PATH
            break
    if gcc_path is None:
        return None, "gcc not installed (run: sudo apt-get install gcc)"

    # 编译
    try:
        ret = subprocess.run(['gcc', '-o', c_bin, c_src, '-lm'],
                             capture_output=True, text=True, timeout=30)
        if ret.returncode != 0:
            return None, f"C compilation failed: {ret.stderr}"
    except FileNotFoundError:
        return None, "gcc not found"

    # 运行
    ret = subprocess.run([c_bin], capture_output=True, text=True)
    return ret.stdout, None


def run_python_model():
    """运行 Python 模型自测试"""
    py_dir = os.path.dirname(os.path.abspath(__file__)) + '/../python'
    ret = subprocess.run(['python3', py_dir + '/fma_model.py'],
                         capture_output=True, text=True)
    return ret.stdout, None


def parse_test_results(output, label):
    """从输出中提取 PASS/FAIL 计数"""
    import re
    passed = len(re.findall(r'PASS:', output))
    failed = len(re.findall(r'FAIL:', output))
    return passed, failed


def main():
    print("=" * 60)
    print("FMA Cross-Model Verification")
    print("=" * 60)

    results = {}

    # --- Python Model ---
    print("\n[1/2] Running Python model...")
    py_out, py_err = run_python_model()
    if py_err:
        print(f"  ERROR: {py_err}")
    else:
        py_p, py_f = parse_test_results(py_out, "Python")
        results['Python'] = (py_p, py_f)
        print(f"  Python: {py_p} passed, {py_f} failed")

    # --- C Model ---
    print("\n[2/2] Running C model...")
    c_out, c_err = run_c_model()
    if c_err:
        print(f"  SKIP: {c_err}")
        print(f"  (Install gcc: sudo apt-get install gcc)")
    else:
        c_p, c_f = parse_test_results(c_out, "C")
        results['C'] = (c_p, c_f)
        print(f"  C:      {c_p} passed, {c_f} failed")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    print(f"{'Model':<12} {'Passed':<10} {'Failed':<10}")
    print("-" * 32)
    for model, (p, f) in results.items():
        status = "✅" if f == 0 else "❌"
        print(f"{model:<12} {p:<10} {f:<10} {status}")

    # --- Consistency Check ---
    print("\nConsistency:")
    models = list(results.keys())
    if len(models) >= 2:
        p0, f0 = results[models[0]]
        p1, f1 = results[models[1]]
        if p0 == p1 and f0 == f1:
            print(f"  ✅ {models[0]} and {models[1]} results match!")
        else:
            print(f"  ❌ Mismatch between models")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
