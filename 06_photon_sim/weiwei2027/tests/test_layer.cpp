// 测试层路径和球体相交组合
#include <iostream>
#include <cmath>

int main() {
    // 模拟参数
    float source_z = -1.0f;
    float brain_start = 1.0f;
    float brain_end = 17.0f;  // 1.0 + 16.0
    float detector_z = 17.5f;
    
    // 测试角度: 0度 (沿z轴)
    float angle_x = 0.0f * M_PI / 180.0f;
    float dx = tanf(angle_x);
    float dz = 1.0f;
    float len = sqrtf(dx*dx + dz*dz);
    dx /= len;
    dz /= len;
    
    // 层入口
    float enter_z = brain_start;
    float exit_z = fminf(detector_z, brain_end);
    float dz_layer = exit_z - enter_z;  // 17 - 1 = 16
    float layer_path_length = dz_layer / fabsf(dz);  // 16 / 0.9848 = 16.25
    
    std::cout << "Angle 0 deg:" << std::endl;
    std::cout << "  dx=" << dx << ", dz=" << dz << std::endl;
    std::cout << "  dz_layer=" << dz_layer << std::endl;
    std::cout << "  layer_path_length=" << layer_path_length << std::endl;
    
    // 层入口位置
    float z_offset = enter_z - source_z;  // 1 - (-1) = 2
    float layer_enter_x = 0 + dx * z_offset / dz;
    std::cout << "  z_offset=" << z_offset << std::endl;
    std::cout << "  layer_enter_x=" << layer_enter_x << std::endl;
    
    // 球体相交 (使用之前的算法结果)
    // t_enter=0, t_exit=8 for center ray
    float t_enter = 0.0f;
    float t_exit = 8.0f;
    
    std::cout << "  Sphere t_enter=" << t_enter << ", t_exit=" << t_exit << std::endl;
    std::cout << "  Check: t_enter < layer_path_length? " << (t_enter < layer_path_length) << std::endl;
    std::cout << "  Check: t_exit > 0? " << (t_exit > 0) << std::endl;
    
    // 段2: 球体内部
    float d2_start = fmaxf(t_enter, 0.0f);
    float d2_end = fminf(t_exit, layer_path_length);
    float d2 = d2_end - d2_start;
    std::cout << "  d2_start=" << d2_start << ", d2_end=" << d2_end << ", d2=" << d2 << std::endl;
    
    // 测试更大角度
    std::cout << "\n---" << std::endl;
    angle_x = 20.0f * M_PI / 180.0f;
    dx = tanf(angle_x);
    dz = 1.0f;
    len = sqrtf(dx*dx + dz*dz);
    dx /= len;
    dz /= len;
    
    layer_enter_x = 0 + dx * 2.0f / dz;
    layer_path_length = 16.0f / fabsf(dz);
    
    std::cout << "Angle 20 deg:" << std::endl;
    std::cout << "  dx=" << dx << ", dz=" << dz << std::endl;
    std::cout << "  layer_enter_x=" << layer_enter_x << std::endl;
    std::cout << "  layer_path_length=" << layer_path_length << std::endl;
    
    // 球体在 z=5, x=0, 半径4
    // 射线: x = layer_enter_x + dx*t, z = 1 + dz*t
    // 在球体处 z=5: t = (5-1)/dz = 4/dz
    // x_at_sphere = layer_enter_x + dx * 4/dz
    float t_to_sphere = (5.0f - 1.0f) / dz;
    float x_at_sphere = layer_enter_x + dx * t_to_sphere;
    float dist_from_center = fabsf(x_at_sphere - 0.0f);
    std::cout << "  t_to_sphere=" << t_to_sphere << std::endl;
    std::cout << "  x_at_sphere=" << x_at_sphere << std::endl;
    std::cout << "  dist_from_center=" << dist_from_center << std::endl;
    std::cout << "  radius=4, hit=" << (dist_from_center < 4.0f) << std::endl;
    
    return 0;
}
