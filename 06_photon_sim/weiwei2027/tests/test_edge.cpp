// 测试边缘角度
#include <iostream>
#include <cmath>

int main() {
    float source_z = -1.0f;
    float brain_start = 1.0f;
    float sphere_z = 5.0f;
    float sphere_r = 4.0f;
    
    // 探测器边界角度
    float det_x = 10.0f;  // 探测器半边宽
    float det_z = 17.5f;
    float max_angle = atanf(det_x / (det_z - source_z));
    std::cout << "Max angle: " << max_angle * 180 / M_PI << " degrees" << std::endl;
    
    // 球体边缘角度 (从源点看)
    // 球体 x 范围: [-4, 4] at z=5
    float sphere_edge_angle_min = atanf(-4.0f / (sphere_z - source_z));
    float sphere_edge_angle_max = atanf(4.0f / (sphere_z - source_z));
    std::cout << "Sphere edge angles: " << sphere_edge_angle_min * 180 / M_PI 
              << " to " << sphere_edge_angle_max * 180 / M_PI << " degrees" << std::endl;
    
    // 测试多个角度
    for (float angle_deg = 0; angle_deg <= 30; angle_deg += 5) {
        float angle = angle_deg * M_PI / 180.0f;
        float dx = tanf(angle);
        float dz = 1.0f;
        float len = sqrtf(dx*dx + dz*dz);
        dx /= len;
        dz /= len;
        
        // 层入口位置 (brain z=1)
        float layer_enter_x = dx * 2.0f / dz;
        
        // 在球体 z=5 处的 x 位置
        float t = (sphere_z - brain_start) / dz;
        float x_at_sphere = layer_enter_x + dx * t;
        
        // 到球体中心距离
        float dist = fabsf(x_at_sphere);
        bool hit = dist < sphere_r;
        
        std::cout << "Angle " << angle_deg << "deg: x_at_sphere=" << x_at_sphere 
                  << ", dist=" << dist << ", hit=" << hit << std::endl;
    }
    
    return 0;
}
