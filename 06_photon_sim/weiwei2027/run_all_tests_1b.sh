#!/bin/bash
# 完整测试脚本 - 支持多平台 (NVIDIA/Iluvatar/MetaX/Moore)
# 默认使用 10亿光子 (1b)
# 作者: weiwei2027
# 日期: 2026-03-15

set -e  # 遇到错误退出

# 平台选择
PLATFORM=${PLATFORM:-nvidia}

echo "========================================"
echo "光子传输模拟 - 完整测试 (10亿光子)"
echo "平台: $PLATFORM"
echo "========================================"
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 根据平台设置可执行文件和命令
case $PLATFORM in
    nvidia)
        GPU_BINARY="./photon_sim_nv"
        CPU_BINARY="./photon_sim_cpu"
        GPU_INFO_CMD="nvidia-smi"
        COMPILER_INFO_CMD="nvcc --version"
        ;;
    metax)
        GPU_BINARY="./photon_sim_metax"
        CPU_BINARY="./photon_sim_cpu"
        GPU_INFO_CMD="echo 'MetaX GPU info not available'"
        COMPILER_INFO_CMD="mxcc --version 2>/dev/null || echo 'mxcc version not available'"
        ;;
    iluvatar)
        GPU_BINARY="./photon_sim_iluvatar"
        CPU_BINARY="./photon_sim_cpu"
        GPU_INFO_CMD="echo 'Iluvatar GPU info not available'"
        COMPILER_INFO_CMD="clang++ --version"
        ;;
    moore)
        GPU_BINARY="./photon_sim_moore"
        CPU_BINARY="./photon_sim_cpu"
        GPU_INFO_CMD="echo 'Moore GPU info not available'"
        COMPILER_INFO_CMD="mcc --version 2>/dev/null || echo 'mcc version not available'"
        ;;
    *)
        echo "错误: 未知平台 $PLATFORM"
        echo "支持的平台: nvidia, metax, iluvatar, moore"
        exit 1
        ;;
esac

echo "测试目录: $SCRIPT_DIR"
echo "平台: $PLATFORM"
echo "光子数量: 10亿 (1,000,000,000)"
echo ""

# 检查可执行文件
if [ ! -f "$GPU_BINARY" ]; then
    echo "错误: 找不到 GPU 可执行文件 $GPU_BINARY"
    echo "请先编译: make PLATFORM=$PLATFORM"
    exit 1
fi

if [ ! -f "$CPU_BINARY" ]; then
    echo "警告: 找不到 CPU 可执行文件 $CPU_BINARY"
    echo "CPU 测试将被跳过"
    HAS_CPU=0
else
    HAS_CPU=1
fi

# 创建输出目录
mkdir -p output/{1layer_gpu,3layer_gpu,3layer_sphere_gpu,parallel_gpu}
if [ $HAS_CPU -eq 1 ]; then
    mkdir -p output/{1layer_cpu,3layer_cpu,3layer_sphere_cpu,parallel_cpu}
fi
mkdir -p output/env_info

echo "=== 1. 记录环境信息 ==="

# 系统信息
echo "--- CPU Information ---" > output/env_info/system_info.txt
echo "CPU Model:" >> output/env_info/system_info.txt
lscpu | grep "Model name" >> output/env_info/system_info.txt 2>/dev/null || echo "Unknown" >> output/env_info/system_info.txt
echo "" >> output/env_info/system_info.txt
echo "CPU Cores: $(nproc)" >> output/env_info/system_info.txt
echo "" >> output/env_info/system_info.txt
echo "Memory:" >> output/env_info/system_info.txt
free -h >> output/env_info/system_info.txt 2>/dev/null || echo "Unknown" >> output/env_info/system_info.txt

# GPU 信息
echo "" >> output/env_info/system_info.txt
echo "--- GPU Information ---" >> output/env_info/system_info.txt
$GPU_INFO_CMD >> output/env_info/system_info.txt 2>/dev/null || echo "GPU info not available" >> output/env_info/system_info.txt

echo "" >> output/env_info/system_info.txt
echo "--- Compiler Version ---" >> output/env_info/system_info.txt
$COMPILER_INFO_CMD >> output/env_info/system_info.txt 2>/dev/null || echo "Compiler info not available" >> output/env_info/system_info.txt

echo "环境信息已保存到 output/env_info/system_info.txt"
cat output/env_info/system_info.txt
echo ""

echo "========================================"
echo "=== GPU 测试 ($PLATFORM) - 10亿光子 ==="
echo "========================================"
echo ""

echo "--- Test 1: 单层几何 (GPU, 1B photons) ---"
$GPU_BINARY \
    -g data/geometry_1layer.txt \
    -m data/materials.csv \
    -s data/source_point_1b.txt \
    -o output/1layer_gpu/
echo ""

echo "--- Test 2: 三层几何 (GPU, 1B photons) ---"
$GPU_BINARY \
    -g data/geometry_3layer.txt \
    -m data/materials.csv \
    -s data/source_point_1b.txt \
    -o output/3layer_gpu/
echo ""

echo "--- Test 3: 三层+球体 (GPU, 1B photons) ---"
$GPU_BINARY \
    -g data/geometry_3layer_sphere.txt \
    -m data/materials.csv \
    -s data/source_point_1b.txt \
    -o output/3layer_sphere_gpu/
echo ""

echo "--- Test 4: 平行束模式 (GPU, 1B photons) ---"
$GPU_BINARY \
    -g data/geometry_3layer.txt \
    -m data/materials.csv \
    -s data/source_parallel_1b.txt \
    -o output/parallel_gpu/
echo ""

if [ $HAS_CPU -eq 1 ]; then
    echo "========================================"
    echo "=== CPU 基线测试 - 10亿光子 ==="
    echo "========================================"
    echo ""
    
    echo "--- Test 5: 单层几何 (CPU, 1B photons) ---"
    $CPU_BINARY \
        -g data/geometry_1layer.txt \
        -m data/materials.csv \
        -s data/source_point_1b.txt \
        -o output/1layer_cpu/
    echo ""
    
    echo "--- Test 6: 三层几何 (CPU, 1B photons) ---"
    $CPU_BINARY \
        -g data/geometry_3layer.txt \
        -m data/materials.csv \
        -s data/source_point_1b.txt \
        -o output/3layer_cpu/
    echo ""
    
    echo "--- Test 7: 三层+球体 (CPU, 1B photons) ---"
    $CPU_BINARY \
        -g data/geometry_3layer_sphere.txt \
        -m data/materials.csv \
        -s data/source_point_1b.txt \
        -o output/3layer_sphere_cpu/
    echo ""
    
    echo "--- Test 8: 平行束模式 (CPU, 1B photons) ---"
    $CPU_BINARY \
        -g data/geometry_3layer.txt \
        -m data/materials.csv \
        -s data/source_parallel_1b.txt \
        -o output/parallel_cpu/
    echo ""
fi

echo "========================================"
echo "=== 测试结果汇总 ==="
echo "========================================"
echo ""

echo "--- GPU 测试结果 (10亿光子) ---"
echo ""
printf "%-30s %-15s %-20s %-15s\n" "Test" "Time (s)" "Rate (photons/s)" "Detected"
printf "%-30s %-15s %-20s %-15s\n" "------------------------------" "---------------" "--------------------" "---------------"

for dir in output/1layer_gpu output/3layer_gpu output/3layer_sphere_gpu output/parallel_gpu; do
    if [ -f "$dir/performance.log" ]; then
        test_name=$(basename $dir)
        time=$(grep "Time:" "$dir/performance.log" | awk '{print $2}')
        rate=$(grep "Processing rate:" "$dir/performance.log" | awk '{print $3}')
        detected=$(grep "Detected:" "$dir/performance.log" | head -1 | awk '{print $2}')
        printf "%-30s %-15s %-20s %-15s\n" "$test_name" "$time" "$rate" "$detected"
    fi
done

if [ $HAS_CPU -eq 1 ]; then
    echo ""
    echo "--- CPU 测试结果 (10亿光子) ---"
    echo ""
    printf "%-30s %-15s %-20s %-15s\n" "Test" "Time (s)" "Rate (photons/s)" "Detected"
    printf "%-30s %-15s %-20s %-15s\n" "------------------------------" "---------------" "--------------------" "---------------"
    
    for dir in output/1layer_cpu output/3layer_cpu output/3layer_sphere_cpu output/parallel_cpu; do
        if [ -f "$dir/performance.log" ]; then
            test_name=$(basename $dir)
            time=$(grep "Time:" "$dir/performance.log" | awk '{print $2}')
            rate=$(grep "Processing rate:" "$dir/performance.log" | awk '{print $3}')
            detected=$(grep "Detected:" "$dir/performance.log" | head -1 | awk '{print $2}')
            printf "%-30s %-15s %-20s %-15s\n" "$test_name" "$time" "$rate" "$detected"
        fi
    done
    
    echo ""
    echo "--- GPU vs CPU 加速比 (10亿光子) ---"
    echo ""
    printf "%-25s %-15s %-15s %-15s\n" "Test" "GPU Time" "CPU Time" "Speedup"
    printf "%-25s %-15s %-15s %-15s\n" "--------------------" "---------------" "---------------" "---------------"
    
    for test in 1layer 3layer 3layer_sphere parallel; do
        if [ -f "output/${test}_gpu/performance.log" ] && [ -f "output/${test}_cpu/performance.log" ]; then
            gpu_time=$(grep "Time:" "output/${test}_gpu/performance.log" | awk '{print $2}')
            cpu_time=$(grep "Time:" "output/${test}_cpu/performance.log" | awk '{print $2}')
            if [ -n "$gpu_time" ] && [ -n "$cpu_time" ] && [ "$gpu_time" != "0" ]; then
                speedup=$(echo "scale=2; $cpu_time / $gpu_time" | bc 2>/dev/null || echo "N/A")
                printf "%-25s %-15s %-15s %-15s\n" "$test" "${gpu_time}s" "${cpu_time}s" "${speedup}x"
            fi
        fi
    done
fi

echo ""
echo "========================================"
echo "测试完成！结果保存在 output/ 目录"
echo "========================================"

# 保存汇总到文件
{
    echo "光子传输模拟 - $PLATFORM 测试汇总 (10亿光子)"
    echo "生成时间: $(date)"
    echo "========================================"
    echo ""
    echo "系统信息:"
    cat output/env_info/system_info.txt
    echo ""
    echo "========================================"
    echo "GPU 测试结果 (10亿光子):"
    for dir in output/1layer_gpu output/3layer_gpu output/3layer_sphere_gpu output/parallel_gpu; do
        if [ -f "$dir/performance.log" ]; then
            echo ""
            echo "--- $(basename $dir) ---"
            cat "$dir/performance.log"
        fi
    done
    
    if [ $HAS_CPU -eq 1 ]; then
        echo ""
        echo "========================================"
        echo "CPU 测试结果 (10亿光子):"
        for dir in output/1layer_cpu output/3layer_cpu output/3layer_sphere_cpu output/parallel_cpu; do
            if [ -f "$dir/performance.log" ]; then
                echo ""
                echo "--- $(basename $dir) ---"
                cat "$dir/performance.log"
            fi
        done
    fi
    
    echo ""
    echo "========================================"
    echo "性能对比:"
    echo ""
    printf "%-25s %-15s %-15s %-15s\n" "Test" "GPU Time" "CPU Time" "Speedup"
    printf "%-25s %-15s %-15s %-15s\n" "--------------------" "---------------" "---------------" "---------------"
    for test in 1layer 3layer 3layer_sphere parallel; do
        if [ -f "output/${test}_gpu/performance.log" ] && [ -f "output/${test}_cpu/performance.log" ]; then
            gpu_time=$(grep "Time:" "output/${test}_gpu/performance.log" | awk '{print $2}')
            cpu_time=$(grep "Time:" "output/${test}_cpu/performance.log" | awk '{print $2}')
            if [ -n "$gpu_time" ] && [ -n "$cpu_time" ] && [ "$gpu_time" != "0" ]; then
                speedup=$(echo "scale=2; $cpu_time / $gpu_time" | bc 2>/dev/null || echo "N/A")
                printf "%-25s %-15s %-15s %-15s\n" "$test" "${gpu_time}s" "${cpu_time}s" "${speedup}x"
            fi
        fi
    done
    
} > output/test_summary_${PLATFORM}_1b.txt

echo ""
echo "详细汇总已保存到: output/test_summary_${PLATFORM}_1b.txt"
