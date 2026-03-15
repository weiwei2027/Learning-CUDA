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
    CXX = nvcc
    CXXFLAGS = -O3 -arch=sm_89 -std=c++17 $(INCLUDES)
    LDFLAGS = -lcuda -lcudart -lcurand
    MAIN_SRC = src/photon_sim_nv.cu
    TARGET = photon_sim_nv
    
else ifeq ($(PLATFORM),iluvatar)
    # Iluvatar CoreX (天数智芯)
    CXX = clang++
    CXXFLAGS = -O3 --cuda-gpu-arch=BI100 -std=c++17 -DILUVATAR_PLATFORM $(INCLUDES)
    LDFLAGS = -lcudart -lcurand
    MAIN_SRC = src/photon_sim_iluvatar.cu
    TARGET = photon_sim_iluvatar
    
else ifeq ($(PLATFORM),metax)
    # MetaX C500 (沐曦)
    CXX = mxcc
    CXXFLAGS = -O3 --cuda-gpu-arch=metax-c500 -std=c++17 -DMETAX_PLATFORM $(INCLUDES)
    LDFLAGS = -lmcudart -lmcrand
    MAIN_SRC = src/photon_sim_metax.maca
    TARGET = photon_sim_metax
    
else ifeq ($(PLATFORM),moore)
    # Moore MTT S5000 (摩尔线程)
    CXX = mcc
    CXXFLAGS = -O3 -arch=mp_21 -std=c++11 -DMOORE_PLATFORM $(INCLUDES)
    LDFLAGS = -lmusart -lmusa_rand
    MAIN_SRC = src/photon_sim_moore.mu
    TARGET = photon_sim_moore
    
else
    $(error Unknown PLATFORM: $(PLATFORM). Valid: nvidia, iluvatar, metax, moore)
endif

# 对象文件
MAIN_OBJ = $(BUILD_DIR)/$(notdir $(MAIN_SRC:.cu=.o))
UTILS_OBJ = $(BUILD_DIR)/utils.o
OBJS = $(MAIN_OBJ) $(UTILS_OBJ)

# ============================================================================
# 构建规则
# ============================================================================

.PHONY: all clean run help

all: $(BUILD_DIR) $(TARGET)

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
	@echo "  all       - Build the target (default)"
	@echo "  clean     - Clean build files"
	@echo "  run       - Build and run with default settings"
	@echo "  help      - Show this help"
	@echo ""
	@echo "Examples:"
	@echo "  make PLATFORM=nvidia"
	@echo "  make PLATFORM=iluvatar run"
	@echo "  make PLATFORM=metax clean"

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
	$(CXX) $(CXXFLAGS) -x cuda -c $< -o $@
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
	$(CXX) $(CXXFLAGS) -x cuda -c $< -o $@
else ifeq ($(PLATFORM),metax)
	$(CXX) $(CXXFLAGS) -x maca -c $< -o $@
else ifeq ($(PLATFORM),moore)
	$(CXX) $(CXXFLAGS) -x musa -c $< -o $@
endif

clean:
	rm -rf build
	rm -f photon_sim_nv photon_sim_iluvatar photon_sim_metax photon_sim_moore
	rm -f $(OUTPUT_DIR)/*.bin $(OUTPUT_DIR)/*.txt $(OUTPUT_DIR)/*.log

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
