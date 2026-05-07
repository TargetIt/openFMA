# openFMA 交付件清单

**项目**: 多精度融合乘加 (FMA) IP 核 (Sky130A, OpenLane v1.1.1)
**设计**: fma_top_stage4 (535 行), 6 种运算模式
**交付日期**: 2026-05-05
**流程状态**: ✅ 全流程通过

---

## 全流程质量门禁

| 门禁项 | 目标 | 实测 | 状态 |
|--------|------|------|------|
| 仿真覆盖率 | 12/12 | 12/12 | ✅ |
| Setup slack (WNS) | >= 0 | 0.0 ns | ✅ |
| Hold slack (WHS) | >= 0 | 0.0 ns | ✅ |
| 关键路径 | < 35ns | 8.24ns | ✅ |
| Clock Period | 35ns | 35ns (28.6 MHz) | ✅ |
| Routing violations | 0 | 0 | ✅ |
| Magic DRC | 0 | 0 | ✅ |
| KLayout DRC | 0 | 0 | ✅ |
| LVS errors | 0 | 0 | ✅ |

---

## Phase 1 — RTL

| 文件 | 说明 |
|------|------|
| `fma_top_stage4.v` | 完整顶层 (FP32/FP16/FP8x4/INT8/Acc16/Acc8x4) |

## Phase 2 — 仿真

| 文件 | 说明 |
|------|------|
| `test_report.md` | 12/12 测试通过 |
| `test_results.log` | 原始仿真输出 |
| `wave.vcd` | VCD 波形 |

## Phase 3 — 综合

| 文件 | 说明 |
|------|------|
| `fma_top_synth.v` | Yosys 综合后门级网表 |
| `fma_top_synth.sdf` | 综合后 SDF |
| `1-synthesis.log` | Yosys 综合日志 |
| `metrics.csv` | 全流程量化指标 |
| `synthesis_report.md` | 签核报告 |

**关键指标**: 8,506 cells, 8.24ns critical path, 0 DRC

## Phase 4 — PnR

| 文件 | 说明 |
|------|------|
| `fma_top_stage4.def` | 最终 DEF |
| `fma_top_stage4.sdc` | 时序约束 |
| `fma_top_pnr.v` | 后 PnR 门级网表 |
| `sdf/` | 多 corner SDF (3x3=9 文件) |
| `spef/` | 多 corner SPEF |
| `pnr_report.md` | 签核报告 |

**关键指标**: 38,540 total cells, 45.86% util, 0 routing vios

## Phase 5 — 物理验证

| 文件 | 说明 |
|------|------|
| `drc.rpt` | Magic DRC: 0 violations |
| `lvs.rpt` | LVS: 0 mismatches |
| `antenna.rpt` | Antenna 违例 |
| `manufacturability.rpt` | 可制造性报告 |
| `verify_report.md` | 签核报告 |

## Phase 6 — GDS

| 文件 | 说明 | 大小 |
|------|------|------|
| `fma_top_stage4.gds` | GDSII 版图 | 23MB |
| `fma_top_stage4.lef` | LEF 抽象视图 | 52KB |
| `fma_top_stage4.lib` | Liberty 时序库 | 132KB |
| `fma_top_stage4.spice` | SPICE 网表 | 2.6MB |
| `gds_report.md` | 签核报告 | - |

---

## 物理设计指标

| 指标 | 值 |
|------|-----|
| 工艺节点 | SkyWater 130nm (sky130A) |
| 时钟频率 | 28.6 MHz (35ns period) |
| 芯片面积 | 0.186 mm² |
| 核心面积 | 171,164 um² |
| 标准单元数 | 8,506 |
| 总单元数 | 38,540 |
| 利用率 | 45.86% |
| 总走线长度 | 262,729 um |
| 过孔数 | 63,571 |
| 峰值内存 | 877 MB |
| 总运行时间 | 13 分 31 秒 |

## 功耗估算

| Corner | Internal | Switching | Leakage |
|--------|----------|-----------|---------|
| Typical | 3.89 nW | 4.41 nW | ~0 nW |

---

## 版图截图 (`images/`)

> 以下图片由 KLayout + Yosys 自动生成

| 图片 | 说明 | 生成工具 |
|------|------|---------|
| `chip_full.png` (1.3MB) | 完整芯片版图 (2400×1800) | KLayout batch mode |
| `final_layout.png` (894KB) | 最终 GDS 版图 (2000×1500) | KLayout batch mode |
| `layout_detail.png` (25KB) | 左下角细节放大 | KLayout batch mode |
| `layout_center.png` (27KB) | 中心区域放大 | KLayout batch mode |
| `synthesis_hierarchy.png` (22MB) | Yosys 综合后层次原理图 | Yosys show + Graphviz dot |
| `synthesis_schematic.dot` (11MB) | 综合后门级 DOT 图 (12,321 cells) | Yosys show |
| `synthesis_hierarchy.dot` (510KB) | 模块层次 DOT 图 | Yosys show |

**设计统计 (来自 Yosys)**:
- 12,321 个标准单元
- 559 个 DFF 触发器
- 2,794 条连线, 20,608 个 wire bits
- 6 种运算模式: FP32/FP16/FP8x4/INT8/FP16→FP32 Acc/FP8x4→FP32 Acc

**物理设计**:
- 425µm × 436µm = 0.186 mm²
- 170 种标准单元类型
- 38,540 总单元数 (含 filler/decap/tap)

---

## 签核结论

**✅ 全流程通过** — 多精度 FMA (FP32/FP16/FP8x4/INT8/Acc) IP 核成功完成 RTL→GDS 实现。所有质量门禁通过 (时序收敛、DRC=0、LVS=0)。GDS/LEF/LIB/SDF/SPEF 等交付件完整可用。

---

*交付件由各 phase 的 gen_report.sh / run_*.sh 自动生成*
*大文件 (GDS/DEF/SDF/SPEF/log) 通过运行脚本重新生成，不存储在 Git 中*
