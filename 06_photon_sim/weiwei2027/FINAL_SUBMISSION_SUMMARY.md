# 最终提交物清单

**项目**: 06 医学成像光子传输模拟  
**作者**: weiwei2027  
**日期**: 2026年3月16日

---

## 📦 提交物结构

```
06_photon_sim/weiwei2027/
├── README.md                    # 项目简介（含效果图）
├── DESIGN.md                    # 设计文档
├── PROJECT_REQUIREMENTS.md      # 赛题要求对照
├── Makefile                     # 多平台构建
├── CMakeLists.txt               # CMake 配置
├── include/                     # 头文件
│   ├── types.h
│   ├── utils.h
│   └── photon_sim.cuh           # 平台抽象层
├── src/                         # 源代码
│   ├── photon_sim_nv.cu         # NVIDIA 版本
│   ├── photon_sim_iluvatar.cu   # Iluvatar 版本
│   ├── photon_sim_metax.maca    # MetaX 版本
│   ├── photon_sim_moore.mu      # Moore 版本
│   ├── photon_sim_cpu.cpp       # CPU 基准
│   └── utils.cpp
├── data/                        # 测试数据
│   ├── geometry_*.txt
│   ├── materials.csv
│   └── source_*.txt
├── scripts/                     # 精简脚本
│   ├── visualize.py
│   ├── visualize_geometry_mpl.py
│   └── generate_report_figures.py
├── report/                      # 技术报告
│   ├── REPORT.md                # 主报告（含图表引用）
│   ├── figures/                 # 生成的图表（6张）
│   │   ├── detector_images.png
│   │   ├── sphere_shadow.png
│   │   ├── performance_comparison.png
│   │   ├── scaling_analysis.png
│   │   ├── geometry_model.png
│   │   └── source_mode_comparison.png
│   ├── archive/                 # 详细测试报告
│   │   ├── A100_BENCHMARK_REPORT.md
│   │   ├── ILUVATAR_BI100_BENCHMARK_REPORT.md
│   │   ├── NCU_ANALYSIS_*.md
│   │   └── TESTING_NOTES.md
│   └── ncu_results/             # NCU原始数据
├── tests/                       # 测试代码
│   ├── test_parser.cpp
│   └── ...
├── results/                     # 测试结果数据
│   ├── A100/
│   ├── BI100/
│   └── C500/
└── archive/                     # 归档的过程文档
    ├── WORKLIST.md
    ├── PORTING_PLAN.md
    └── ...
```

---

## ✅ 核心提交物

| 文件/目录 | 说明 | 必须 |
|-----------|------|------|
| `README.md` | 项目简介，含效果图 | ✅ |
| `DESIGN.md` | 物理模型与算法设计 | ✅ |
| `PROJECT_REQUIREMENTS.md` | 赛题要求对照 | ✅ |
| `report/REPORT.md` | 技术报告（主报告） | ✅ |
| `report/figures/` | 图表（6张） | ✅ |
| `src/` | 源代码（4平台+CPU） | ✅ |
| `include/` | 头文件 | ✅ |
| `Makefile` | 多平台构建 | ✅ |
| `data/` | 测试数据 | ✅ |

---

## 📊 生成的图表

| 图表 | 文件 | 用途 |
|------|------|------|
| 探测器图像 | `detector_images.png` | 展示三层几何和球体效果 |
| 球体阴影 | `sphere_shadow.png` | 展示血块阴影效果 |
| 性能对比 | `performance_comparison.png` | 多平台对比 |
| 扩展性分析 | `scaling_analysis.png` | RTX 4090 不同规模测试 |
| 几何模型 | `geometry_model.png` | 几何模型示意图 |
| 光源模式对比 | `source_mode_comparison.png` | 点源 vs 平行束（含球体对比） |

---

## 🔧 编译测试

```bash
# 清理并编译 NVIDIA 版本
make clean && make PLATFORM=nvidia

# 运行基础测试
./photon_sim_nv -g data/geometry_3layer.txt -m data/materials.csv -s data/source_point_10m.txt -o output/

# 完整测试（10亿光子，推荐）
./run_all_tests_1b.sh

# 可视化结果
cd scripts && python3 visualize.py ../output/image.bin --info ../output/image_info.txt
```

### 测试脚本说明

| 脚本 | 用途 |
|------|------|
| `run_all_tests_1b.sh` | **主要测试脚本**，运行4种配置（单层/三层/球体/平行束）× 10亿光子 |
| `scripts/generate_report_figures.py` | 生成报告图表 |
| `scripts/visualize.py` | 可视化探测器图像 |

---

## 📄 PDF 报告转换

使用 **VS Code + Markdown PDF 插件**：

1. 安装 Markdown PDF 插件
2. 打开 `report/REPORT.md`
3. 右键 → Markdown PDF: Export (pdf)

> 注：已使用此方法生成 `report/REPORT.pdf`

---

## 🎯 关键指标

| 指标 | 数值 |
|------|------|
| 最高处理速率 | 1.11×10¹⁰ photons/sec |
| GPU 加速比 | ~1175× (vs CPU) |
| 测试平台 | 3个（NVIDIA、Iluvatar、MetaX）|
| 穿透率误差 | < 0.1% |

---

**提交状态**: ✅ 已准备就绪
