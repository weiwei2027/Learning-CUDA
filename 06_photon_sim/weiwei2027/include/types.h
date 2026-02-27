#ifndef TYPES_H
#define TYPES_H

// 检测是否在 CUDA 环境中编译
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#else
// CPU 编译时的替代定义
#include <cstddef>
#endif

// 区域类型
enum RegionType {
    LAYER = 0,
    SPHERE = 1
};

// 几何区域结构
struct Region {
    RegionType type;
    char name[32];
    float thickness;    // layer: 厚度, sphere: 半径
    float x, y, z;      // 起始点/球心坐标
    int material_id;    // 材料索引
    char material_name[32]; // 材料名称
};

// 材料属性
struct Material {
    char name[32];
    float mu;           // 线性衰减系数 (cm^-1)
    float energy;       // 能量 (keV)
};

// 光子结构
struct Photon {
    float x, y, z;      // 位置
    float dx, dy, dz;   // 方向（单位向量）
    float energy;       // 能量
    float path_left;    // 剩余自由程
    bool alive;         // 是否存活
    int region_id;      // 当前所在区域
};

// 源参数
struct Source {
    int type;           // 0: point, 1: parallel
    float x, y, z;      // 源位置
    float dx, dy, dz;   // 主方向
    float energy;       // 光子能量
    int num_photons;    // 光子总数
};

// 探测器参数
struct Detector {
    float x, y, z;      // 探测器位置
    float width, height; // 尺寸 (cm)
    int nx, ny;         // 像素数
    float *pixels;      // 像素计数数组 (device)
};

// 模拟参数
struct SimParams {
    int num_regions;
    int num_materials;
    int num_photons;
    float detector_z;
    float world_min[3];
    float world_max[3];
};

#endif // TYPES_H
