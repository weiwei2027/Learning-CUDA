// 验证角度采样是否覆盖整个探测器
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>

int main() {
    // 配置参数
    float source_x = 0.0f, source_y = 0.0f, source_z = -1.0f;
    float detector_z = 17.5f, detector_width = 20.0f, detector_height = 20.0f;
    float detector_x = 0.0f, detector_y = 0.0f;
    
    float dz = detector_z - source_z;  // 18.5
    
    // 探测器边界
    float det_x_min = detector_x - detector_width / 2.0f;   // -10
    float det_x_max = detector_x + detector_width / 2.0f;   // +10
    float det_y_min = detector_y - detector_height / 2.0f;  // -10
    float det_y_max = detector_y + detector_height / 2.0f;  // +10
    
    // 角度范围
    float angle_x_min = atanf((det_x_min - source_x) / dz);
    float angle_x_max = atanf((det_x_max - source_x) / dz);
    float angle_y_min = atanf((det_y_min - source_y) / dz);
    float angle_y_max = atanf((det_y_max - source_y) / dz);
    
    std::cout << "=== 角度范围计算 ===" << std::endl;
    std::cout << "dz = " << dz << std::endl;
    std::cout << "det_x range: [" << det_x_min << ", " << det_x_max << "]" << std::endl;
    std::cout << "angle_x range: [" << angle_x_min * 180/M_PI << ", " << angle_x_max * 180/M_PI << "] deg" << std::endl;
    
    // 验证边界
    float tan_min = tanf(angle_x_min);
    float tan_max = tanf(angle_x_max);
    float x_at_min = source_x + dz * tan_min;
    float x_at_max = source_x + dz * tan_max;
    
    std::cout << "\n=== 边界验证 ===" << std::endl;
    std::cout << "At angle_x_min: tan = " << tan_min << ", x = " << x_at_min << std::endl;
    std::cout << "At angle_x_max: tan = " << tan_max << ", x = " << x_at_max << std::endl;
    std::cout << "Expected x range: [" << det_x_min << ", " << det_x_max << "]" << std::endl;
    
    // 采样测试
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist_angle_x(angle_x_min, angle_x_max);
    std::uniform_real_distribution<float> dist_angle_y(angle_y_min, angle_y_max);
    
    float min_x = 1e10, max_x = -1e10;
    float min_y = 1e10, max_y = -1e10;
    
    for (int i = 0; i < 100000; i++) {
        float angle_x = dist_angle_x(gen);
        float angle_y = dist_angle_y(gen);
        
        // 计算方向（归一化）
        float tan_x = tanf(angle_x);
        float tan_y = tanf(angle_y);
        float len = sqrtf(tan_x*tan_x + tan_y*tan_y + 1.0f);
        float dx = tan_x / len;
        float dy = tan_y / len;
        float dz_dir = 1.0f / len;
        
        // 计算到达探测器的位置
        // source_z + t * dz_dir = detector_z
        float t = (detector_z - source_z) / dz_dir;
        float hit_x = source_x + t * dx;
        float hit_y = source_y + t * dy;
        
        min_x = std::min(min_x, hit_x);
        max_x = std::max(max_x, hit_x);
        min_y = std::min(min_y, hit_y);
        max_y = std::max(max_y, hit_y);
    }
    
    std::cout << "\n=== 采样测试 (100k photons) ===" << std::endl;
    std::cout << "Actual hit x range: [" << min_x << ", " << max_x << "]" << std::endl;
    std::cout << "Actual hit y range: [" << min_y << ", " << max_y << "]" << std::endl;
    std::cout << "Expected x range: [" << det_x_min << ", " << det_x_max << "]" << std::endl;
    
    // 检查是否覆盖
    bool covers_x = (min_x <= det_x_min + 0.1f) && (max_x >= det_x_max - 0.1f);
    bool covers_y = (min_y <= det_y_min + 0.1f) && (max_y >= det_y_max - 0.1f);
    
    std::cout << "\n=== 覆盖检查 ===" << std::endl;
    std::cout << "X coverage: " << (covers_x ? "OK" : "FAIL") << std::endl;
    std::cout << "Y coverage: " << (covers_y ? "OK" : "FAIL") << std::endl;
    
    return 0;
}
