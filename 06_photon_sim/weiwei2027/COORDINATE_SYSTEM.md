# 医学成像光子传输模拟 - 坐标系设计规范

> 文档版本: 1.0  
> 作者: weiwei2027  
> 日期: 2026-02-28

---

## 1. 坐标系定义原则

### 1.1 右手坐标系

```
         Y (垂直向上)
         ↑
         │
         │    Z (深度方向，光子传播)
         │   ╱
         │  ╱
         │ ╱
         └──────────→ X (水平向右)
        原点(0,0,0)
```

### 1.2 关键设计决策

| 决策项 | 当前选择 | 替代方案 | 建议 |
|--------|----------|----------|------|
| 原点位置 | 皮肤层起点 (0,0,0) | 探测器中心 | 保持当前 |
| Z轴方向 | 从源到探测器 | 反向 | 保持当前 |
| XY范围 | 以Z轴为中心 | 任意偏移 | 保持居中 |

---

## 2. 各组件坐标规范

### 2.1 X射线源

```yaml
# 推荐配置
position = 0.0 0.0 -d_source    # d_source = 源到皮肤距离，通常 1-10 cm
```

**要求**: 源必须在 Z 轴上（X=0, Y=0），确保对称性。

### 2.2 探测器

```yaml
# 推荐配置
detector_position = 0.0 0.0 z_detector    # Z轴上，z_detector > 所有层厚之和
detector_size = width height              # 覆盖所有可能的投影范围
```

**关键规则**: 探测器中心必须在 Z 轴上（X=0, Y=0）。

### 2.3 几何层（平板）

```yaml
# 格式: layer name thickness x y z material
# x, y 必须为 0.0（居中）
layer skin 0.2 0.0 0.0 0.0 tissue_skin
layer skull 0.8 0.0 0.0 0.2 tissue_bone
```

**要求**: 所有层的 X=0, Y=0，沿 Z 轴堆叠。

### 2.4 球体异物

```yaml
# 格式: sphere name radius cx cy cz material
# 球心 XY 必须在探测器覆盖范围内
sphere hematoma 0.5 0.0 0.0 5.0 tissue_blood  # ✅ 在光路中心
sphere tumor 0.3 5.0 5.0 8.0 tissue_blood     # ⚠️ 需要发散角支持
```

**约束**: 球体投影必须在探测器范围内，即：
```
|cx| + radius < detector_width / 2
|cy| + radius < detector_height / 2
```

---

## 3. 光源发散模型

### 3.1 当前实现

#### 3.1.1 Point 模式（点源）

**特点**: 光子从固定点出发，有位置偏移

```cpp
// 点源：位置偏移 [-2, 2] cm，方向沿Z轴
float r1 = dist(gen);  // [-1, 1]
float r2 = dist(gen);  // [-1, 1]
photon.x = r1 * 2.0f;  // [-2, 2] cm
photon.y = r2 * 2.0f;  // [-2, 2] cm
photon.z = source.z;   // 源平面
photon.dx = 0; photon.dy = 0; photon.dz = 1;
```

**有效覆盖**: 探测器中心区域 ±2 cm

#### 3.1.2 Parallel 模式（平行束）

**特点**: 光子在源平面上均匀分布，方向平行

```cpp
// 平行束：位置分布 [-10, 10] cm，方向固定
float x = dist_pos(gen);  // [-10, 10] cm
float y = dist_pos(gen);  // [-10, 10] cm
photon.x = x;
photon.y = y;
photon.z = source.z;
photon.dx = source.dx;    // 使用源方向
photon.dy = source.dy;
photon.dz = source.dz;
```

**有效覆盖**: 整个探测器范围 ±10 cm

**注意**: 当前两种模式结果相同，因为都是垂直Z轴直线传播。

### 3.2 完整发散模型（待实现）

#### 方案A: 锥形束（Cone Beam）

光子从点源发出，呈锥形发散。

```cpp
// 采样锥形束内的方向
float theta_max = atan(detector_size / (2 * source_distance));
float theta = theta_max * sqrt(random());  // 均匀采样
float phi = 2 * M_PI * random();

direction.x = sin(theta) * cos(phi);
direction.y = sin(theta) * sin(phi);
direction.z = cos(theta);
```

**适用**: CT扫描仪模拟

#### 方案B: 朗伯分布（Lambertian）

符合物理的余弦分布，更真实。

```cpp
// 朗伯分布采样
float cos_theta = sqrt(random());  // 余弦加权
direction.z = cos_theta;
float sin_theta = sqrt(1 - cos_theta * cos_theta);
float phi = 2 * M_PI * random();
direction.x = sin_theta * cos(phi);
direction.y = sin_theta * sin(phi);
```

**适用**: 真实X射线管模拟

#### 方案C: 平行束（Parallel Beam）

所有光子方向平行，常用于简化模型。

```cpp
// 固定方向，仅位置随机
direction = (0, 0, 1);
position.x = random_in_range(-field_size/2, field_size/2);
position.y = random_in_range(-field_size/2, field_size/2);
```

**适用**: 远距离源近似（如同步辐射）

---

## 4. 比赛参数的坐标系问题

### 4.1 当前问题

比赛示例 `sphere hematoma 0.5 5.0 5.0 5.0` 存在问题：

| 问题 | 说明 |
|------|------|
| XY偏移过大 | (5,5) 距离中心 7.07 cm |
| 光源发散不足 | 当前仅覆盖 [-2,2] cm |
| 结果 | 球体永远不被照射 |

### 4.2 解决方案

#### 方案1: 修改球体位置（当前采用）

将球体移到光路中心：

```
sphere hematoma 0.5 0.0 0.0 5.0 tissue_blood
```

**优点**: 立即验证球体功能  
**缺点**: 偏离比赛示例

#### 方案2: 增大光源发散角

实现锥形束或朗伯分布，确保能照射到 (5,5)。

**需要调整**:
- 探测器尺寸可能需要增大到 30x30 cm 或更大
- 实现完整的方向采样算法

**验证公式**:
```
最大偏移角 θ = atan(5 / (5 - (-1))) = atan(5/6) ≈ 40°
需要锥形束半角 > 40°
探测器半宽 > 17.5 * tan(40°) ≈ 14.7 cm
```

---

## 5. 推荐配置

### 5.1 Point 模式配置

```yaml
# source.txt
source_type = point
position = 0.0 0.0 -1.0
direction = 0.0 0.0 1.0
num_photons = 10000000
detector_position = 0.0 0.0 17.5
detector_size = 20.0 20.0
detector_pixels = 1024 1024

# geometry.txt
layer skin 0.2 0.0 0.0 0.0 tissue_skin
layer skull 0.8 0.0 0.0 0.2 tissue_bone
layer brain 16.0 0.0 0.0 1.0 tissue_brain
sphere hematoma 0.5 0.0 0.0 5.0 tissue_blood  # 中心位置
```

### 5.2 Parallel 模式配置

```yaml
# source_parallel.txt
source_type = parallel
position = 0.0 0.0 -1.0          # 源平面位置
direction = 0.0 0.0 1.0          # 光束方向
num_photons = 10000000
detector_position = 0.0 0.0 17.5
detector_size = 20.0 20.0
detector_pixels = 1024 1024

# geometry.txt (与 point 模式相同)
layer skin 0.2 0.0 0.0 0.0 tissue_skin
layer skull 0.8 0.0 0.0 0.2 tissue_bone
layer brain 16.0 0.0 0.0 1.0 tissue_brain
sphere hematoma 0.5 0.0 0.0 5.0 tissue_blood
```

### 5.3 测试命令

```bash
# Point 模式
cd cmake-build-debug/06_photon_sim/weiwei2027
./photon_sim_cpu -g data/geometry.txt -s data/source.txt -o output/

# Parallel 模式
./photon_sim_cpu -g data/geometry.txt -s data/source_parallel.txt -o output/
```

### 5.2 完整配置（待实现发散角后）

```yaml
# source.txt - 锥形束
source_type = cone_beam
position = 0.0 0.0 -50.0       # 远距离源
divergence_angle = 20.0        # 半角 20°
num_photons = 10000000
detector_position = 0.0 0.0 17.5
detector_size = 40.0 40.0      # 更大探测器
detector_pixels = 1024 1024

# geometry.txt
layer skin 0.2 0.0 0.0 0.0 tissue_skin
layer skull 0.8 0.0 0.0 0.2 tissue_bone
layer brain 16.0 0.0 0.0 1.0 tissue_brain
sphere hematoma 0.5 5.0 5.0 5.0 tissue_blood  # 偏离中心的球体
```

---

## 6. 坐标系验证清单

在提交前检查：

- [ ] 光源位置 X=0, Y=0
- [ ] 探测器中心 X=0, Y=0
- [ ] 所有层的 X=0, Y=0
- [ ] 球体投影在探测器范围内
- [ ] 球体 Z 坐标在某层内
- [ ] 光源发散角能覆盖球体位置

---

## 7. 结论

1. **当前实现**: 使用中心球体 `(0,0,5)`，简化光源模型
2. **未来改进**: 实现完整发散角模型，支持偏离中心的异物
3. **报告说明**: 在报告中注明光源模型的假设和限制

**关键洞察**: 比赛示例的 `(5,5,5)` 位置隐含要求实现**完整的光源发散模型**，这是比球体处理本身更复杂的功能。
