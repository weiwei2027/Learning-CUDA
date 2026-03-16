# 测试目录

## test_parser.cpp

**目的**: 验证输入文件解析模块的正确性

**测试内容**:
1. **几何文件解析** (`geometry_*.txt`)
   - 解析 LAYER/SPHERE 区域定义
   - 验证区域连续性和厚度计算

2. **材料文件解析** (`materials.csv`)
   - CSV 表头识别和数据行解析
   - 不同能量下 μ 值的精确匹配和线性插值

3. **源参数解析** (`source_*.txt`)
   - 点源/平行束模式识别
   - 位置、方向、能量、光子数解析
   - 探测器配置读取

**编译运行**:
```bash
make        # 编译并运行测试
make test   # 同上
make clean  # 清理编译产物
```

**依赖**: `../src/utils.cpp`, `../include/utils.h`, `../include/types.h`
