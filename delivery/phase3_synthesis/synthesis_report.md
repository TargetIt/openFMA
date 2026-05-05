# Phase 3 FMA 综合交付件报告

**生成时间**: 2026-05-05 12:12:00
**运行目录**: RUN_2026.05.05_03.57.29
**流程状态**: flow completed

## 综合结果摘要

| 指标 | 实测值 | 判定 |
|------|--------|------|
| 标准单元数 | 0 | - |
| 总单元数 (含 filler) | 239 | - |
| 关键路径延迟 | 659ns | ✅ 时序收敛 |
| Die Area | 96747.28095184325mm^2 | - |
| Core Area | 1905um^2 | - |
| Final Utilization | 45.8633% | - |
| WNS | 0.0ns | ✅ |
| TNS | 0.0ns | ✅ |
| 综合 ERROR | 1 | ❌ |
| 综合 WARNING | 21 | - |
| 总运行时间 | 0h13m31s0ms | - |

## 综合流程步骤

| 步骤 | 说明 | 日志 |
|------|------|------|
| Linter | Verilator lint | logs/synthesis/linter.log |
| Synthesis | Yosys 综合 | logs/synthesis/1-synthesis.log |
| STA | 单 corner 时序 | logs/synthesis/2-sta.log |

## 交付件清单

| 文件 | 说明 |
|------|------|
| 综合后网表 | results/synthesis/fma_top_stage4.v |
| SDF | results/synthesis/fma_top_stage4.sdf |
| Metrics CSV | reports/metrics.csv |

## 综合评审签核

| 检查项 | 状态 |
|--------|------|
| RTL 源代码已就绪 | ✅ |
| 综合流程完整运行 | ✅ |
| 无时序违例 | ✅ |
| 综合无 ERROR | ❌ |

**综合评审结论**: ⚠️ 需检查

---

*此报告由 gen_report.sh 自动生成*
