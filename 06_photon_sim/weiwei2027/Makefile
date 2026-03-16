# 医学成像光子传输模拟 Makefile
# 支持多平台: NVIDIA, Iluvatar, MetaX, Moore

# 默认平台
PLATFORM ?= nvidia

# 包含目录
INCLUDES = -I./include

# 构建目录
BUILD_DIR = build/$(PLATFORM)

# 输出目录
OUTPUT_DIR = output

# 通用源文件
UTILS_SRC = src/utils.cpp

# ============================================================================
# 平台特定配置
# ============================================================================

ifeq ($(PLATFORM),nvidia)
    # NVIDIA CUDA
    # 支持多架构: A100 (sm_80), RTX 4090 (sm_89)
    CXX = nvcc
    CXXFLAGS = -O3 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -std=c++17 $(INCLUDES)
    LDFLAGS = -lcuda -lcudart -lcurand
    MAIN_SRC = src/photon_sim_nv.cu
    TARGET = photon_sim_nv
    
else ifeq ($(PLATFORM),iluvatar)
    # Iluvatar CoreX (天数智芯)
    CXX = clang++
    CXXFLAGS = -std=c++17 -O3 -DPLATFORM_ILUVATAR $(INCLUDES) -I/usr/local/corex/include -fPIC
    LDFLAGS = -lcudart -L/usr/local/corex/lib64
    MAIN_SRC = src/photon_sim_iluvatar.cu
    TARGET = photon_sim_iluvatar
    
else ifeq ($(PLATFORM),metax)
    # MetaX C500 (沐曦)
    CXX = mxcc
    CXXFLAGS = -std=c++17 -O3 -DMETAX_PLATFORM $(INCLUDES) -I/opt/maca/include
    LDFLAGS = -L/opt/maca/lib
    MAIN_SRC = src/photon_sim_metax.maca
    TARGET = photon_sim_metax
    
else ifeq ($(PLATFORM),moore)
    # Moore MTT S5000 (摩尔线程)
    CXX = mcc
    CXXFLAGS = -std=c++11 -O3 -DPLATFORM_MOORE $(INCLUDES) -I/usr/local/musa/include -fPIC
    LDFLAGS = -L/usr/lib/gcc/x86_64-linux-gnu/11/ -L/usr/local/musa/lib -lmusart
    MAIN_SRC = src/photon_sim_moore.mu
    TARGET = photon_sim_moore
    
else
    $(error Unknown PLATFORM: $(PLATFORM). Valid: nvidia, iluvatar, metax, moore)
endif

# 对象文件
# 对象文件（支持 .cu, .maca, .mu 等多种扩展名）
MAIN_OBJ = $(BUILD_DIR)/$(basename $(notdir $(MAIN_SRC))).o
UTILS_OBJ = $(BUILD_DIR)/utils.o
OBJS = $(MAIN_OBJ) $(UTILS_OBJ)

# CPU 版本
CPU_TARGET = photon_sim_cpu
CPU_SRC = src/photon_sim_cpu.cpp
CPU_OBJ = $(BUILD_DIR)/photon_sim_cpu.o
CPU_UTILS_OBJ = $(BUILD_DIR)/utils_cpu.o

# ============================================================================
# 构建规则
# ============================================================================

.PHONY: all clean run help

# 默认目标: 同时编译 GPU 和 CPU 版本
all: $(BUILD_DIR) $(TARGET) $(CPU_TARGET)

# 只编译 GPU 版本
gpu_only: $(BUILD_DIR) $(TARGET)

help:
	@echo "Usage: make PLATFORM=<platform> [target]"
	@echo ""
	@echo "Platforms:"
	@echo "  nvidia    - NVIDIA GPU (nvcc, default)"
	@echo "  iluvatar  - 天数智芯 CoreX BI-V100 (clang++)"
	@echo "  metax     - 沐曦 C500 (mxcc)"
	@echo "  moore     - 摩尔线程 MTT S5000 (mcc)"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build both GPU and CPU (default)"
	@echo "  photon_sim_cpu - Build CPU baseline"
	@echo "  gpu_only     - Build only GPU version"
	@echo "  clean        - Clean build files"
	@echo "  run          - Build and run GPU version"
	@echo "  run_cpu      - Build and run CPU version"
	@echo "  help         - Show this help"
	@echo ""
	@echo "Examples:"
	@echo "  make                    # Build both GPU and CPU (default)"
	@echo "  make gpu_only           # Build only GPU version"
	@echo "  make photon_sim_cpu     # Build only CPU version"
	@echo "  make run                # Build and run GPU"
	@echo "  make run_cpu            # Build and run CPU"

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# 主程序编译规则
ifeq ($(PLATFORM),nvidia)
$(MAIN_OBJ): $(MAIN_SRC)
	$(CXX) $(CXXFLAGS) -c $< -o $@
else ifeq ($(PLATFORM),iluvatar)
$(MAIN_OBJ): $(MAIN_SRC)
	$(CXX) $(CXXFLAGS) -c $< -o $@
else ifeq ($(PLATFORM),metax)
$(MAIN_OBJ): $(MAIN_SRC)
	$(CXX) $(CXXFLAGS) -x maca -c $< -o $@
else ifeq ($(PLATFORM),moore)
$(MAIN_OBJ): $(MAIN_SRC)
	$(CXX) $(CXXFLAGS) -x musa -c $< -o $@
endif

# utils.cpp 编译规则
$(UTILS_OBJ): $(UTILS_SRC) | $(BUILD_DIR)
ifeq ($(PLATFORM),nvidia)
	$(CXX) $(CXXFLAGS) -x cu -c $< -o $@
else ifeq ($(PLATFORM),iluvatar)
	$(CXX) $(CXXFLAGS) -c $< -o $@
else ifeq ($(PLATFORM),metax)
	$(CXX) $(CXXFLAGS) -x maca -c $< -o $@
else ifeq ($(PLATFORM),moore)
	$(CXX) $(CXXFLAGS) -x musa -c $< -o $@
endif

# ============================================================================
# CPU 基准版本
# ============================================================================

$(CPU_TARGET): $(CPU_OBJ) $(CPU_UTILS_OBJ)
	g++ -O3 -std=c++17 -o $@ $^ $(INCLUDES)

$(CPU_OBJ): $(CPU_SRC) | $(BUILD_DIR)
	g++ -O3 -std=c++17 -c $< -o $@ $(INCLUDES)

$(CPU_UTILS_OBJ): $(UTILS_SRC) | $(BUILD_DIR)
	g++ -O3 -std=c++17 -c $< -o $@ $(INCLUDES)

# CPU 运行目标
run_cpu: $(CPU_TARGET)
	./$(CPU_TARGET) -g data/geometry_3layer.txt -m data/materials.csv \
		-s data/source_point_10m.txt -o $(OUTPUT_DIR)/

# 同时编译 GPU 和 CPU 版本
all_with_cpu: all $(CPU_TARGET)

clean:
	rm -rf build
	rm -f photon_sim_nv photon_sim_iluvatar photon_sim_metax photon_sim_moore
	rm -f $(CPU_TARGET)
	rm -f $(OUTPUT_DIR)/*.bin $(OUTPUT_DIR)/*.txt $(OUTPUT_DIR)/*.log 2>/dev/null || true

run: all
	./$(TARGET) -g data/geometry_3layer.txt -m data/materials.csv \
		-s data/source_point_10m.txt -o $(OUTPUT_DIR)/

# 测试不同配置
test_1layer: all
	./$(TARGET) -g data/geometry_1layer.txt -m data/materials.csv \
		-s data/source_point_10m.txt -o $(OUTPUT_DIR)/

test_sphere: all
	./$(TARGET) -g data/geometry_3layer_sphere.txt -m data/materials.csv \
		-s data/source_point_10m.txt -o $(OUTPUT_DIR)/
