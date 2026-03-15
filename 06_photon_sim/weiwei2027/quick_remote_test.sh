#!/bin/bash
# 快速远程测试脚本 - 只运行一个测试用于快速验证
# 用法: ./quick_remote_test.sh [1layer|3layer|sphere|parallel]

TEST_TYPE=${1:-"3layer"}  # 默认测试三层几何

# 配置
REMOTE_USER="weiwei"
REMOTE_HOST="8.145.51.96"
REMOTE_PORT="2222"
KEY_PATH="${KEY_PATH:-$HOME/workspace/InfiniTensor2025/ssh/nvidia/key.id}"
REMOTE_WORK_DIR="/home/weiwei/weiwei2027"

echo "快速远程测试 - $TEST_TYPE"
echo "============================"

# 检查密钥
if [ ! -f "$KEY_PATH" ]; then
    echo "错误: 找不到密钥文件"
    exit 1
fi

chmod 600 "$KEY_PATH"

# 根据测试类型选择参数
case $TEST_TYPE in
    1layer)
        GEOMETRY="data/geometry_1layer.txt"
        OUTPUT="output/quick_1layer"
        ;;
    3layer)
        GEOMETRY="data/geometry_3layer.txt"
        OUTPUT="output/quick_3layer"
        ;;
    sphere)
        GEOMETRY="data/geometry_3layer_sphere.txt"
        OUTPUT="output/quick_sphere"
        ;;
    parallel)
        GEOMETRY="data/geometry_3layer.txt"
        SOURCE="data/source_parallel_10m.txt"
        OUTPUT="output/quick_parallel"
        ;;
    *)
        echo "未知测试类型: $TEST_TYPE"
        echo "可用选项: 1layer, 3layer, sphere, parallel"
        exit 1
        ;;
esac

SOURCE=${SOURCE:-"data/source_point_10m.txt"}

echo "测试配置:"
echo "  几何: $GEOMETRY"
echo "  光源: $SOURCE"
echo "  输出: $OUTPUT"
echo ""

# 上传代码
echo "上传代码..."
rsync -avz -e "ssh -p $REMOTE_PORT -i $KEY_PATH -o StrictHostKeyChecking=no" \
    --exclude='build/' --exclude='cmake-build-*/' --exclude='output/' --exclude='.git/' \
    --exclude='photon_sim_nv' --exclude='photon_sim_cpu' \
    --exclude='*.o' --exclude='*.exe' \
    ./ "$REMOTE_USER@$REMOTE_HOST:$REMOTE_WORK_DIR/" > /dev/null 2>&1

# 执行测试
ssh -p $REMOTE_PORT -i "$KEY_PATH" -o StrictHostKeyChecking=no \
    "$REMOTE_USER@$REMOTE_HOST" "
        cd $REMOTE_WORK_DIR
        make clean >/dev/null 2>&1
        make PLATFORM=nvidia >/dev/null 2>&1
        mkdir -p $OUTPUT
        ./photon_sim_nv -g $GEOMETRY -m data/materials.csv -s $SOURCE -o $OUTPUT/
        cat $OUTPUT/performance.log
    "

echo ""
echo "测试完成!"
