#!/bin/bash
# 远程 GPU 平台一键测试脚本
# 使用 rsync 传输，支持在作品目录内独立构建和测试

set -e

# 配置
REMOTE_USER="weiwei"
REMOTE_HOST="8.145.51.96"
REMOTE_PORT="2222"
KEY_PATH="${KEY_PATH:-$HOME/workspace/InfiniTensor2025/ssh/nvidia/key.id}"
REMOTE_WORK_DIR="/home/weiwei/weiwei2027"
LOCAL_RESULT_DIR="./remote_results_$(date +%Y%m%d_%H%M%S)"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  远程 GPU 平台一键测试脚本${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查密钥
if [ ! -f "$KEY_PATH" ]; then
    echo -e "${RED}错误: 找不到密钥文件 $KEY_PATH${NC}"
    echo "请设置环境变量: export KEY_PATH=/path/to/key.id"
    exit 1
fi

chmod 600 "$KEY_PATH"

# 检查当前目录
if [ ! -f "Makefile" ] || [ ! -f "CMakeLists.txt" ]; then
    echo -e "${RED}错误: 当前目录不是项目根目录${NC}"
    echo "请在 06_photon_sim/weiwei2027/ 目录下执行此脚本"
    exit 1
fi

echo "配置信息:"
echo "  远程服务器: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PORT"
echo "  远程工作目录: $REMOTE_WORK_DIR"
echo "  本地结果目录: $LOCAL_RESULT_DIR"
echo ""

# Step 1: 上传到远程
echo -e "${YELLOW}=== Step 1: 上传代码到远程服务器 ===${NC}"
rsync -avz --progress \
    -e "ssh -p $REMOTE_PORT -i $KEY_PATH -o StrictHostKeyChecking=no" \
    --exclude='build' \
    --exclude='cmake-build-*' \
    --exclude='output/*.bin' \
    --exclude='output/*.log' \
    --exclude='photon_sim_*' \
    --exclude='.git' \
    --exclude='*.o' \
    --exclude='*.tar.gz' \
    --exclude='remote_results_*' \
    ./ \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_WORK_DIR/"

if [ $? -ne 0 ]; then
    echo -e "${RED}上传失败!${NC}"
    exit 1
fi

echo -e "${GREEN}上传完成!${NC}"
echo ""

# Step 2: 在远程执行构建和测试
echo -e "${YELLOW}=== Step 2: 远程构建和测试 ===${NC}"
echo "即将在远程服务器执行测试，需要约 5-10 分钟..."
echo "按回车继续，或 Ctrl+C 取消..."
read -t 3 || true

# 创建远程测试脚本
REMOTE_SCRIPT=$(cat << 'REMOTE_EOF'
#!/bin/bash
set -e

cd /home/weiwei/weiwei2027

echo "========================================"
echo "远程服务器环境信息"
echo "========================================"
echo ""
echo "--- CPU 信息 ---"
lscpu | grep "Model name" || echo "未知"
echo "核心数: $(nproc)"
echo ""
echo "--- GPU 信息 ---"
nvidia-smi || echo "nvidia-smi 不可用"
echo ""
echo "--- CUDA 版本 ---"
nvcc --version || echo "nvcc 不可用"
echo ""

# 创建输出目录
mkdir -p output/{1layer_gpu,3layer_gpu,3layer_sphere_gpu,parallel_gpu}
mkdir -p output/env_info

# 保存环境信息
cat > output/env_info/system_info.txt << INFO_EOF
CPU: $(lscpu | grep "Model name" | cut -d: -f2 | xargs)
Cores: $(nproc)
Memory: $(free -h | grep Mem | awk '{print $2}')
GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "N/A")
CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $5}' || echo "N/A")
INFO_EOF

echo "========================================"
echo "构建项目"
echo "========================================"
make clean 2>/dev/null || true
make PLATFORM=nvidia

echo ""
echo "========================================"
echo "运行测试 (10亿光子)"
echo "========================================"

echo ""
echo "--- Test 1: 单层几何 (GPU, 1B photons) ---"
./photon_sim_nv -g data/geometry_1layer.txt -m data/materials.csv -s data/source_point_1b.txt -o output/1layer_gpu/

echo ""
echo "--- Test 2: 三层几何 (GPU, 1B photons) ---"
./photon_sim_nv -g data/geometry_3layer.txt -m data/materials.csv -s data/source_point_1b.txt -o output/3layer_gpu/

echo ""
echo "--- Test 3: 三层+球体 (GPU, 1B photons) ---"
./photon_sim_nv -g data/geometry_3layer_sphere.txt -m data/materials.csv -s data/source_point_1b.txt -o output/3layer_sphere_gpu/

echo ""
echo "--- Test 4: 平行束模式 (GPU, 1B photons) ---"
./photon_sim_nv -g data/geometry_3layer.txt -m data/materials.csv -s data/source_parallel_1b.txt -o output/parallel_gpu/

echo ""
echo "========================================"
echo "测试结果汇总"
echo "========================================"
echo ""
printf "%-30s %-15s %-20s\n" "Test" "Time (s)" "Rate (photons/s)"
printf "%-30s %-15s %-20s\n" "------------------------------" "---------------" "--------------------"

for dir in output/1layer_gpu output/3layer_gpu output/3layer_sphere_gpu output/parallel_gpu; do
    if [ -f "$dir/performance.log" ]; then
        test_name=$(basename $dir)
        time=$(grep "Time:" "$dir/performance.log" | awk '{print $2}')
        rate=$(grep "Processing rate:" "$dir/performance.log" | awk '{print $3}')
        printf "%-30s %-15s %-20s\n" "$test_name" "$time" "$rate"
    fi
done

echo ""
echo "测试完成!"
REMOTE_EOF
)

# 执行远程脚本
ssh -p $REMOTE_PORT -i "$KEY_PATH" -o StrictHostKeyChecking=no \
    "$REMOTE_USER@$REMOTE_HOST" "$REMOTE_SCRIPT"

if [ $? -ne 0 ]; then
    echo -e "${RED}远程测试失败!${NC}"
    exit 1
fi

echo -e "${GREEN}远程测试完成!${NC}"
echo ""

# Step 3: 下载结果
echo -e "${YELLOW}=== Step 3: 下载测试结果 ===${NC}"
mkdir -p "$LOCAL_RESULT_DIR"

rsync -avz --progress \
    -e "ssh -p $REMOTE_PORT -i $KEY_PATH -o StrictHostKeyChecking=no" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_WORK_DIR/output/" \
    "$LOCAL_RESULT_DIR/"

if [ $? -ne 0 ]; then
    echo -e "${RED}下载结果失败!${NC}"
    exit 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  所有测试完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "结果保存在: $LOCAL_RESULT_DIR"
echo ""

# 显示汇总
if [ -f "$LOCAL_RESULT_DIR/env_info/system_info.txt" ]; then
    echo "环境信息:"
    cat "$LOCAL_RESULT_DIR/env_info/system_info.txt"
    echo ""
fi

echo "性能日志:"
for log in $LOCAL_RESULT_DIR/*/performance.log; do
    if [ -f "$log" ]; then
        test_name=$(basename $(dirname $log))
        rate=$(grep "Processing rate:" "$log" | awk '{print $3}')
        echo "  $test_name: $rate photons/sec"
    fi
done

echo ""
echo -e "${GREEN}测试报告已保存!${NC}"
