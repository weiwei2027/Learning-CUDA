# 配置文件说明

本目录包含光子传输模拟所需的所有输入配置文件。

## 目录

- [光源配置文件](#光源配置文件)
- [几何模型文件](#几何模型文件)
- [材料数据文件](#材料数据文件)
- [快速开始](#快速开始)
- [典型测试场景](#典型测试场景)

---

## 光源配置文件

光源配置分为**点光源 (point)** 和**平行光源 (parallel)** 两种模式，每种模式提供三个数量级：

| 文件名 | 光源类型 | 光子数量 | 适用场景 |
|--------|---------|---------|---------|
| `source_point_10m.txt` | 点光源 (锥形束) | 1,000万 (10⁷) | 快速功能验证 |
| `source_point_100m.txt` | 点光源 (锥形束) | 1亿 (10⁸) | 标准性能测试 |
| `source_point_1b.txt` | 点光源 (锥形束) | 10亿 (10⁹) | 高精度统计验证 |
| `source_parallel_10m.txt` | 平行光源 | 1,000万 (10⁷) | 快速功能验证 |
| `source_parallel_100m.txt` | 平行光源 | 1亿 (10⁸) | 标准性能测试 |
| `source_parallel_1b.txt` | 平行光源 | 10亿 (10⁹) | 高精度统计验证 |

### 点光源 vs 平行光源

- **点光源 (point)**: 从固定点发射锥形束，覆盖探测器视场
  - 用于模拟传统 CT 点源照射
  - 光子分布在探测器上呈不均匀分布（中心密集）

- **平行光源 (parallel)**: 整个源平面均匀发射平行光
  - 用于模拟理想平行束照射
  - 光子均匀分布在探测器上
  - **推荐用于物理验证**（透过率计算更简单）

---

## 几何模型文件

几何模型按复杂度分为三种：

| 文件名 | 层数 | 球体异物 | 用途 |
|--------|------|---------|------|
| `geometry_1layer.txt` | 1层 (皮肤) | 无 | Beer-Lambert 定律验证 |
| `geometry_3layer.txt` | 3层 (皮肤/颅骨/脑组织) | 无 | 多层介质传输测试 |
| `geometry_3layer_sphere.txt` | 3层 (皮肤/颅骨/脑组织) | 有 (血块) | 球体相交算法测试 |

### 几何模型详解

#### geometry_1layer.txt
单层皮肤组织，厚度 1.0 cm
- **用途**: 基础物理验证
- **预期透过率**: $T = e^{-0.2 \times 1.0} \approx 81.87\%$

#### geometry_3layer.txt
模拟头部三层结构：
- 皮肤层: 0.2 cm (μ = 0.2 cm⁻¹) → 光学厚度 0.04
- 颅骨层: 0.8 cm (μ = 0.5 cm⁻¹) → 光学厚度 0.40  
- 脑组织层: 16.0 cm (μ = 0.18 cm⁻¹) → 光学厚度 2.88
- **总光学厚度**: 0.04 + 0.40 + 2.88 = **3.32**
- **预期透过率**: $T = e^{-3.32} \approx 3.62\%$

#### geometry_3layer_sphere.txt
在 geometry_3layer 基础上增加：
- 血块异物: 半径 2.0 cm，位于 (0, 0, 5.0)，材料为血液 (μ = 0.22 cm⁻¹)
- **用途**: 验证球体-光线相交算法，球体遮挡效应应在探测器图像中心形成明显阴影
- **注意**: 由于只有穿过球体的光线会被额外衰减，整体透过率难以简单估算

---

## 材料数据文件

### materials.csv

包含四种生物组织的线性衰减系数 μ (cm⁻¹)：

| 材料 | 50 keV | 100 keV |
|------|--------|---------|
| tissue_skin (皮肤) | 0.20 | 0.15 |
| tissue_bone (骨骼) | 0.50 | 0.30 |
| tissue_brain (脑组织) | 0.18 | 0.12 |
| tissue_blood (血液) | 0.22 | 0.16 |

**注意**: 程序会根据源的能量自动选择对应衰减系数（支持 50-100 keV 线性插值）

---

## 快速开始

### 基础功能验证（最快）

```bash
# 单层 + 平行光 + 1000万光子
./photon_sim -g data/geometry_1layer.txt -s data/source_parallel_10m.txt

# 验证透过率是否接近 81.87%
```

### 标准性能测试

```bash
# 3层头部 + 平行光 + 1亿光子
./photon_sim -g data/geometry_3layer.txt -s data/source_parallel_100m.txt
```

### 球体功能测试

```bash
# 3层 + 球体 + 1000万光子
./photon_sim -g data/geometry_3layer_sphere.txt -s data/source_point_10m.txt
```

### 极限性能测试

```bash
# 3层 + 平行光 + 10亿光子
./photon_sim -g data/geometry_3layer.txt -s data/source_parallel_1b.txt
```

---

## 典型测试场景

| 测试目的 | 推荐配置 | 预期结果 |
|---------|---------|---------|
| Beer-Lambert 验证 | `geometry_1layer.txt` + `source_parallel_10m.txt` | 透过率 ≈ 81.87% |
| GPU/CPU 结果一致性 | `geometry_3layer.txt` + `source_point_10m.txt` | 差异 < 0.1% |
| 球体相交功能 | `geometry_3layer_sphere.txt` + `source_parallel_10m.txt` | 球体区域有阴影 |
| 性能基准测试 | `geometry_1layer.txt` + `source_parallel_1b.txt` | 记录 photons/sec |
| 医学成像场景 | `geometry_3layer.txt` + `source_point_100m.txt` | 模拟真实 CT |

---

## 文件命名规则

### 光源文件
```
source_{type}_{scale}.txt
  │       │      └─ 10m / 100m / 1b (光子数量)
  │       └─ point / parallel (光源类型)
  └─ 固定前缀
```

### 几何文件
```
geometry_{layers}[_sphere].txt
  │        │        └─ 可选：是否包含球体异物
  │        └─ 1layer / 3layer (层数)
  └─ 固定前缀
```
