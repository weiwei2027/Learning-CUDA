# 远程测试快速开始

## 环境准备

设置环境变量（添加到 `~/.bashrc`）：

```bash
export KEY_PATH="$HOME/workspace/InfiniTensor2025/ssh/nvidia/key.id"
chmod 600 $KEY_PATH
```

## 三种测试方式

### 方式 1: 一键完整测试（推荐）

```bash
cd 06_photon_sim/weiwei2027
./run_remote_tests.sh
```

自动完成：上传 → 构建 → 测试 → 下载结果

结果保存在：`remote_results_YYYYMMDD_HHMMSS/`

### 方式 2: 快速单个测试

```bash
# 测试三层几何（约2分钟）
./quick_remote_test.sh 3layer

# 其他选项
./quick_remote_test.sh 1layer    # 单层
./quick_remote_test.sh sphere    # 带球体
./quick_remote_test.sh parallel  # 平行束
```

### 方式 3: 手动测试

```bash
# 1. 上传代码
rsync -avz -e "ssh -p 2222 -i $KEY_PATH" \
    ./ weiwei@8.145.51.96:/home/weiwei/weiwei2027/

# 2. SSH 登录
ssh -p 2222 -i $KEY_PATH weiwei@8.145.51.96

# 3. 远程构建和测试
cd /home/weiwei/weiwei2027
make clean && make PLATFORM=nvidia
./photon_sim_nv -g data/geometry_3layer.txt -m data/materials.csv -s data/source_point_10m.txt -o output/

# 4. 下载结果
rsync -avz -e "ssh -p 2222 -i $KEY_PATH" \
    weiwei@8.145.51.96:/home/weiwei/weiwei2027/output/ ./remote_results/
```

## 远程目录结构

上传后，代码位于：`/home/weiwei/weiwei2027/`

```
/home/weiwei/weiwei2027/
├── CMakeLists.txt          # 独立构建配置
├── Makefile                # 备用构建方式
├── src/                    # 源代码
├── data/                   # 测试数据
├── output/                 # 测试结果
└── build/                  # 构建输出（自动生成）
```

## 常见问题

### Q: rsync 连接失败？
```bash
# 检查密钥权限
chmod 600 $KEY_PATH

# 测试 SSH 连接
ssh -p 2222 -i $KEY_PATH weiwei@8.145.51.96
```

### Q: make 命令找不到？
```bash
# 确保在正确目录
pwd  # 应该显示 /home/weiwei/weiwei2027

# 检查 Makefile 存在
ls -la Makefile
```

### Q: CUDA 编译错误？
```bash
# 检查 CUDA 环境
which nvcc
nvcc --version

# 设置环境变量
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## 详细文档

完整指南见：[remote_test_guide.md](remote_test_guide.md)
