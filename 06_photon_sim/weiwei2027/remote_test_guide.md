# 远程平台测试指南

## 概述

本文档描述如何在远程 GPU 平台（A100/Iluvatar/MetaX/Moore）上测试光子传输模拟项目。

**重要**: 所有操作都在 `06_photon_sim/weiwei2027/` 目录内进行，不依赖项目根目录。

---

## 目录结构

```
06_photon_sim/weiwei2027/
├── CMakeLists.txt          # 独立 CMake 配置
├── Makefile                # 备用构建方式
├── src/                    # 源代码
├── include/                # 头文件
├── data/                   # 测试数据
├── build/                  # 构建输出目录（本地）
└── output/                 # 测试结果目录
```

---

## 前置条件

### 1. 远程服务器信息

| 平台 | 服务器 | 端口 | 密钥路径 |
|------|--------|------|----------|
| NVIDIA A100 | weiwei@8.145.51.96 | 2222 | ~/workspace/InfiniTensor2025/ssh/nvidia/key.id |

### 2. 本地环境变量（添加到 ~/.bashrc）

```bash
# 项目路径
export PROJECT_DIR="$HOME/workspace/InfiniTensor2025/weiwei2027/Learning-CUDA/06_photon_sim/weiwei2027"
export KEY_PATH="$HOME/workspace/InfiniTensor2025/ssh/nvidia/key.id"
export REMOTE_HOST="weiwei@8.145.51.96"
export REMOTE_PORT="2222"
```

---

## 快速开始

### 步骤 1: 上传到远程服务器

```bash
cd $PROJECT_DIR

# 使用 rsync 上传（推荐）
# 注意：不要排除 'photon_sim_*'，这会排除掉 src/ 目录下的源代码！
rsync -avz --progress \
    -e "ssh -p $REMOTE_PORT -i $KEY_PATH" \
    --exclude='build/' \
    --exclude='cmake-build-*/' \
    --exclude='output/*.bin' \
    --exclude='output/*.log' \
    --exclude='photon_sim_nv' \
    --exclude='photon_sim_cpu' \
    --exclude='photon_sim_iluvatar' \
    --exclude='photon_sim_metax' \
    --exclude='photon_sim_moore' \
    --exclude='*.o' \
    --exclude='.git/' \
    ./ \
    $REMOTE_HOST:/home/weiwei/weiwei2027/
```

### 步骤 2: SSH 登录并构建

```bash
ssh -p $REMOTE_PORT -i $KEY_PATH $REMOTE_HOST

cd /home/weiwei/weiwei2027

# 构建 NVIDIA 版本
make clean
make PLATFORM=nvidia

# 或 CMake 方式
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build --parallel
```

### 步骤 3: 运行测试

```bash
# 创建输出目录
mkdir -p output/{1layer,3layer,3layer_sphere,parallel}

# 测试 1: 单层几何
./photon_sim_nv \
    -g data/geometry_1layer.txt \
    -m data/materials.csv \
    -s data/source_point_10m.txt \
    -o output/1layer/

# 测试 2: 三层几何
./photon_sim_nv \
    -g data/geometry_3layer.txt \
    -m data/materials.csv \
    -s data/source_point_10m.txt \
    -o output/3layer/

# 测试 3: 三层+球体
./photon_sim_nv \
    -g data/geometry_3layer_sphere.txt \
    -m data/materials.csv \
    -s data/source_point_10m.txt \
    -o output/3layer_sphere/

# 测试 4: 平行束模式
./photon_sim_nv \
    -g data/geometry_3layer.txt \
    -m data/materials.csv \
    -s data/source_parallel_10m.txt \
    -o output/parallel/
```

### 步骤 4: 下载结果到本地

```bash
# 在本地执行
rsync -avz --progress \
    -e "ssh -p $REMOTE_PORT -i $KEY_PATH" \
    $REMOTE_HOST:/home/weiwei/weiwei2027/output/ \
    ./remote_results/
```

---

## 多平台构建

### NVIDIA A100

```bash
make PLATFORM=nvidia
./photon_sim_nv -g data/geometry_3layer.txt -m data/materials.csv -s data/source_point_10m.txt -o output/
```

### Iluvatar CoreX (天数智芯)

```bash
make PLATFORM=iluvatar
./photon_sim_iluvatar -g data/geometry_3layer.txt -m data/materials.csv -s data/source_point_10m.txt -o output/
```

### MetaX C500 (沐曦)

```bash
make PLATFORM=metax
./photon_sim_metax -g data/geometry_3layer.txt -m data/materials.csv -s data/source_point_10m.txt -o output/
```

### Moore MTT S5000 (摩尔线程)

```bash
make PLATFORM=moore
./photon_sim_moore -g data/geometry_3layer.txt -m data/materials.csv -s data/source_point_10m.txt -o output/
```

---

## 一键测试脚本

使用 `run_remote_tests.sh` 脚本自动完成上传、测试、下载：

```bash
# 在本地执行
./run_remote_tests.sh
```

脚本会自动：
1. 使用 rsync 上传代码到远程
2. SSH 登录并执行所有测试
3. 下载结果到本地 `remote_results/` 目录
4. 生成测试报告

---

## 故障排除

### 问题 1: rsync 连接失败

**症状**: `rsync: Connection closed`

**解决**: 检查密钥权限
```bash
chmod 600 $KEY_PATH
```

### 问题 2: make 找不到目标

**症状**: `make: *** No rule to make target...`

**解决**: 确保在正确目录
```bash
pwd  # 应该是 /home/weiwei/weiwei2027
ls Makefile  # 确认文件存在
```

### 问题 3: CUDA 编译错误

**症状**: `nvcc not found`

**解决**: 检查 CUDA 环境
```bash
which nvcc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 问题 4: 数据文件找不到

**症状**: `Error: Failed to parse geometry file`

**解决**: 检查 data 目录
```bash
ls data/
# 如果不存在，手动复制
rsync -avz -e "ssh -p $REMOTE_PORT -i $KEY_PATH" data/ $REMOTE_HOST:/home/weiwei/weiwei2027/data/
```

---

## 性能对比参考

| 平台 | GPU | 处理速率 (photons/sec) | 备注 |
|------|-----|----------------------|------|
| NVIDIA | RTX 4090 | 4.23×10¹⁰ | 本地测试 |
| NVIDIA | A100 | 待测试 | 远程测试目标 |
| Iluvatar | BI-V100 | 待测试 | 代码就绪 |
| MetaX | C500 | 待测试 | 代码就绪 |
| Moore | MTT S5000 | 待测试 | 代码就绪 |

---

## 联系信息

- 作者: weiwei2027
- 项目: 06 医学成像光子传输模拟
- 训练营: 2025冬季训练营 CUDA方向
