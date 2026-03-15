// 验证光子初始化逻辑
#include <iostream>
#include <cmath>
#include <random>

int main() {
    // 参数
    float source_x = 0.0f, source_y = 0.0f, source_z = -1.0f;
    float detector_z = 17.5f;
    float detector_x = 0.0f, detector_width = 20.0f;
    float detector_y = 0.0f, detector_height = 20.0f;
    
    // 计算锥形束角度范围
    float dz = detector_z - source_z;  // 18.5
    float det_x_min = detector_x - detector_width / 2.0f;   // -10
    float det_x_max = detector_x + detector_width / 2.0f;   // +10
    float det_y_min = detector_y - detector_height / 2.0f;  // -10
    float det_y_max = detector_y + detector_height / 2.0f;  // +10
    
    float angle_x_min = atanf((det_x_min - source_x) / dz);  // atan(-10/18.5)
    float angle_x_max = atanf((det_x_max - source_x) / dz);  // atan(+10/18.5)
    
    std::cout << "Detector half-angle X: " << angle_x_min * 180/M_PI 
              << " to " << angle_x_max * 180/M_PI << " degrees" << std::endl;
    
    // 随机数生成器
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist_angle_x(angle_x_min, angle_x_max);
    std::uniform_real_distribution<float> dist_angle_y(atanf(-10/18.5), atanf(10/18.5));
    
    // 统计穿过球体的光子数
    int hit_sphere = 0;
    int total = 100000;
    
    for (int i = 0; i < total; i++) {
        float angle_x = dist_angle_x(gen);
        float angle_y = dist_angle_y(gen);
        
        float tan_x = tanf(angle_x);
        float tan_y = tanf(angle_y);
        float len = sqrtf(tan_x*tan_x + tan_y*tan_y + 1.0f);
        float dx = tan_x / len;
        float dy = tan_y / len;
        float dz_dir = 1.0f / len;
        
        // 计算层入口位置 (brain z=1)
        float brain_start = 1.0f;
        float z_offset = brain_start - source_z;  // 2
        float layer_enter_x = source_x + dx * z_offset / dz_dir;
        float layer_enter_y = source_y + dy * z_offset / dz_dir;
        
        // 检查是否穿过球体 (0,0,5), r=4
        // 计算在球体 z=5 处的位置
        float t_to_sphere = (5.0f - brain_start) / dz_dir;
        float x_at_sphere = layer_enter_x + dx * t_to_sphere;
        float y_at_sphere = layer_enter_y + dy * t_to_sphere;
        float dist_from_center = sqrtf(x_at_sphere*x_at_sphere + y_at_sphere*y_at_sphere);
        
        if (dist_from_center < 4.0f) {
            hit_sphere++;
        }
    }
    
    std::cout << "\nOut of " << total << " photons:" << std::endl;
    std::cout << "  Hit sphere: " << hit_sphere << " (" << 100.0*hit_sphere/total << "%)" << std::endl;
    std::cout << "  Miss sphere: " << total - hit_sphere << " (" << 100.0*(total-hit_sphere)/total << "%)" << std::endl;
    
    // 理论值: 球体立体角 / 探测器立体角
    // 简化: 球体截面积 / 探测器距离处的截面积
    float sphere_area = M_PI * 4.0f * 4.0f;  // 50.27 cm^2
    float dist_to_sphere = 6.0f;  // 5 - (-1)
    float sphere_solid_angle = sphere_area / (dist_to_sphere * dist_to_sphere);  // 球体近似
    
    std::cout << "\nApproximate sphere solid angle: " << sphere_solid_angle << " sr" << std::endl;
    
    return 0;
}
