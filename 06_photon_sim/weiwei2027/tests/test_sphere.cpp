// 测试球体相交算法
#include <iostream>
#include <cmath>

struct Photon {
    float x, y, z;
    float dx, dy, dz;
};

bool intersectSphere(
    float px, float py, float pz,
    float dx, float dy, float dz,
    float cx, float cy, float cz, float radius,
    float& t_enter, float& t_exit)
{
    float ox = cx - px;
    float oy = cy - py;
    float oz = cz - pz;

    float b = -2.0f * (dx * ox + dy * oy + dz * oz);
    float c = ox * ox + oy * oy + oz * oz - radius * radius;
    float discriminant = b * b - 4.0f * c;

    if (discriminant < 0.0f) return false;

    float sqrt_disc = sqrtf(discriminant);
    t_enter = (-b - sqrt_disc) * 0.5f;
    t_exit = (-b + sqrt_disc) * 0.5f;

    if (t_exit < 0.0f) return false;
    if (t_enter < 0.0f) t_enter = 0.0f;

    return true;
}

int main() {
    // 测试1: 沿z轴穿过球体中心的光子
    // 球体: center=(0,0,5), r=4
    // 起点: (0,0,1), 方向: (0,0,1)
    float t_enter, t_exit;
    bool hit = intersectSphere(0, 0, 1, 0, 0, 1, 0, 0, 5, 4, t_enter, t_exit);
    std::cout << "Test 1 (center): hit=" << hit << ", t_enter=" << t_enter << ", t_exit=" << t_exit << std::endl;
    
    // 测试2: 斜向穿过球体的光子
    // 起点: (2, 0, 1), 方向需要计算
    float dx = 0, dy = 0, dz = 1; // 沿z轴
    hit = intersectSphere(2, 0, 1, dx, dy, dz, 0, 0, 5, 4, t_enter, t_exit);
    std::cout << "Test 2 (offset x=2): hit=" << hit << ", t_enter=" << t_enter << ", t_exit=" << t_exit << std::endl;
    
    // 测试3: 刚好擦过球体边缘
    // 距离中心4cm的点
    hit = intersectSphere(4, 0, 1, dx, dy, dz, 0, 0, 5, 4, t_enter, t_exit);
    std::cout << "Test 3 (edge x=4): hit=" << hit << ", t_enter=" << t_enter << ", t_exit=" << t_exit << std::endl;
    
    // 测试4: 未击中球体
    hit = intersectSphere(5, 0, 1, dx, dy, dz, 0, 0, 5, 4, t_enter, t_exit);
    std::cout << "Test 4 (miss x=5): hit=" << hit << std::endl;
    
    // 测试5: 使用实际锥形束角度
    // 角度 x = 10度 = 0.1745 rad
    float angle_x = 10.0f * M_PI / 180.0f;
    float angle_y = 0;
    dx = tanf(angle_x);
    dy = tanf(angle_y);
    dz = 1.0f;
    float len = sqrtf(dx*dx + dy*dy + dz*dz);
    dx /= len; dy /= len; dz /= len;
    std::cout << "\nTest 5 (10 deg angle): dx=" << dx << ", dz=" << dz << std::endl;
    hit = intersectSphere(0, 0, 1, dx, dy, dz, 0, 0, 5, 4, t_enter, t_exit);
    std::cout << "  hit=" << hit << ", t_enter=" << t_enter << ", t_exit=" << t_exit << std::endl;
    
    return 0;
}
