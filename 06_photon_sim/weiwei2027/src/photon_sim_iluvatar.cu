/**
 * @file photon_sim_nv.cpp
 * @brief NVIDIA CUDA 光子传输模拟 - 平台特定实现
 * @author weiwei2027
 * 
 * 对应平台: NVIDIA GPU (A100, RTX系列等)
 * 编译器: nvcc
 * 运行时: CUDA Runtime
 */

#include <iostream>
#include <fstream>
#include <chrono>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../include/types.h"
#include "../include/utils.h"

// ============================================================================
// CUDA 错误检查宏
// ============================================================================
// 使用 photon_sim.cuh 中定义的 GPU_CHECK

// ============================================================================
// 常量内存定义
// ============================================================================
#define MAX_REGIONS 16
#define MAX_SPHERES 8

__device__ __constant__ Region d_const_regions[MAX_REGIONS];
__device__ __constant__ float d_const_mu[MAX_REGIONS];
__device__ __constant__ int d_const_num_regions;
__device__ __constant__ int d_const_num_spheres;
__device__ __constant__ int d_const_sphere_indices[MAX_SPHERES];

// ============================================================================
// 设备函数
// ============================================================================

__device__ __forceinline__ float sampleFreePath(float mu, gpuRandState *state)
{
    if (mu <= 0.0f) return 1e30f;
    float xi = gpuRandUniform(state);
    if (xi < 1e-7f) xi = 1e-7f;
    return -logf(xi) / mu;
}

__device__ __forceinline__ bool intersectSphere(
    float px, float py, float pz,
    float dx, float dy, float dz,
    float cx, float cy, float cz, float radius,
    float &t_enter, float &t_exit)
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

__device__ __forceinline__ bool processLayerWithSpheres(
    Photon &p, int layer_idx,
    float layer_enter_z, float layer_path_length,
    gpuRandState &state)
{
    float z_offset = layer_enter_z - p.z;
    float layer_enter_x = p.x + p.dx * z_offset / (p.dz + 1e-10f);
    float layer_enter_y = p.y + p.dy * z_offset / (p.dz + 1e-10f);

    bool hit_sphere = false;

    for (int s = 0; s < d_const_num_spheres && !hit_sphere; s++) {
        int sphere_idx = d_const_sphere_indices[s];
        const Region &sphere = d_const_regions[sphere_idx];

        float t_enter, t_exit;
        if (intersectSphere(layer_enter_x, layer_enter_y, layer_enter_z,
                           p.dx, p.dy, p.dz,
                           sphere.x, sphere.y, sphere.z, sphere.thickness,
                           t_enter, t_exit)) {

            if (t_enter < layer_path_length && t_exit > 0.0f) {
                hit_sphere = true;

                // 段1: 层起点到球体入口
                float d1 = fminf(t_enter, layer_path_length);
                if (d1 > 0.0f) {
                    if (sampleFreePath(d_const_mu[layer_idx], &state) < d1)
                        return true;
                }

                // 段2: 球体内部
                float d2 = fminf(t_exit, layer_path_length) - fmaxf(t_enter, 0.0f);
                if (d2 > 0.0f) {
                    if (sampleFreePath(d_const_mu[sphere_idx], &state) < d2)
                        return true;
                }

                // 段3: 球体出口到层终点
                float d3 = layer_path_length - fminf(t_exit, layer_path_length);
                if (d3 > 0.0f) {
                    if (sampleFreePath(d_const_mu[layer_idx], &state) < d3)
                        return true;
                }
            }
        }
    }

    if (!hit_sphere) {
        if (sampleFreePath(d_const_mu[layer_idx], &state) < layer_path_length)
            return true;
    }

    p.x += p.dx * layer_path_length;
    p.y += p.dy * layer_path_length;
    return false;
}

// ============================================================================
// 核函数
// ============================================================================

__global__ void initRandState(gpuRandState *state, int n, unsigned long seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) gpuRandInit(seed, tid, 0, &state[tid]);
}

__global__ void clearDetectorKernel(float *pixels, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) pixels[tid] = 0.0f;
}

__global__ void photonTransportKernel(
    Detector detector,
    SimParams params,
    gpuRandState *rand_states,
    Source source)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    gpuRandState localState = rand_states[tid];

    // 预计算探测器边界
    float det_x_min = detector.x - detector.width / 2.0f;
    float det_x_max = detector.x + detector.width / 2.0f;
    float det_y_min = detector.y - detector.height / 2.0f;
    float det_y_max = detector.y + detector.height / 2.0f;

    // 预计算点源角度范围
    float dz = detector.z - source.z;
    float angle_x_min = atanf((det_x_min - source.x) / dz);
    float angle_x_max = atanf((det_x_max - source.x) / dz);
    float angle_y_min = atanf((det_y_min - source.y) / dz);
    float angle_y_max = atanf((det_y_max - source.y) / dz);

    // 查找第一层起始位置
    float first_layer_start = 1e30f;
    for (int i = 0; i < d_const_num_regions; i++) {
        if (d_const_regions[i].type == LAYER) {
            first_layer_start = fminf(first_layer_start, d_const_regions[i].z);
        }
    }

    // 每个线程处理多个光子 (grid-stride)
    for (int i = tid; i < params.num_photons; i += total_threads) {
        Photon p;

        // 光源初始化
        if (source.type == 1) {  // 平行光源
            p.x = det_x_min + curand_uniform(&localState) * (det_x_max - det_x_min);
            p.y = det_y_min + curand_uniform(&localState) * (det_y_max - det_y_min);
            p.z = source.z;
            p.dx = source.dx;
            p.dy = source.dy;
            p.dz = source.dz;
        } else {  // 点光源
            p.x = source.x;
            p.y = source.y;
            p.z = source.z;

            float angle_x = angle_x_min + curand_uniform(&localState) * (angle_x_max - angle_x_min);
            float angle_y = angle_y_min + curand_uniform(&localState) * (angle_y_max - angle_y_min);
            float tan_x = tanf(angle_x);
            float tan_y = tanf(angle_y);
            float len = sqrtf(tan_x * tan_x + tan_y * tan_y + 1.0f);

            p.dx = tan_x / len;
            p.dy = tan_y / len;
            p.dz = 1.0f / len;
        }
        p.energy = source.energy;
        p.alive = true;

        float current_z = p.z;
        bool absorbed = false;

        // 传播到第一层入口
        if (current_z < first_layer_start && p.dz > 0.0f) {
            float path_to_first = (first_layer_start - current_z) / p.dz;
            p.x += p.dx * path_to_first;
            p.y += p.dy * path_to_first;
            current_z = first_layer_start;
        }

        // 逐层传输
        for (int r = 0; r < d_const_num_regions && !absorbed; r++) {
            if (d_const_regions[r].type != LAYER) continue;

            float layer_start = d_const_regions[r].z;
            float layer_end = layer_start + d_const_regions[r].thickness;

            if (current_z >= layer_end) continue;

            float enter_z = fmaxf(current_z, layer_start);
            float exit_z = fminf(params.detector_z, layer_end);
            if (exit_z <= enter_z) continue;

            float layer_path_length = (exit_z - enter_z) / (p.dz + 1e-10f);
            absorbed = processLayerWithSpheres(p, r, enter_z, layer_path_length, localState);
            current_z = exit_z;
        }

        // 传播到探测器
        if (!absorbed && current_z < detector.z) {
            float path_to_detector = (detector.z - current_z) / (p.dz + 1e-10f);
            p.x += p.dx * path_to_detector;
            p.y += p.dy * path_to_detector;
        }

        // 记录到探测器
        if (!absorbed) {
            int px = (int)((p.x + detector.width / 2.0f) / detector.width * detector.nx);
            int py = (int)((p.y + detector.height / 2.0f) / detector.height * detector.ny);
            px = max(0, min(px, detector.nx - 1));
            py = max(0, min(py, detector.ny - 1));

            gpuAtomicAdd(&detector.pixels[py * detector.nx + px], 1.0f);
        }
    }

    rand_states[tid] = localState;
}

// ============================================================================
// 主机辅助函数
// ============================================================================

void prepareSphereIndices(const Region *regions, int num_regions,
                          int *sphere_indices, int &num_spheres)
{
    num_spheres = 0;
    for (int i = 0; i < num_regions && num_spheres < MAX_SPHERES; i++) {
        if (regions[i].type == SPHERE) {
            sphere_indices[num_spheres++] = i;
        }
    }
}

void copyGeometryToConstant(const Region *h_regions, const float *h_mu,
                            int num_regions, cudaStream_t stream = 0)
{
    int sphere_indices[MAX_SPHERES];
    int num_spheres;
    prepareSphereIndices(h_regions, num_regions, sphere_indices, num_spheres);

    GPU_CHECK(gpuMemcpyToSymbolAsync(d_const_regions, h_regions,
                                       num_regions * sizeof(Region), 0,
                                       gpuMemcpyHostToDevice, stream));
    GPU_CHECK(gpuMemcpyToSymbolAsync(d_const_mu, h_mu,
                                       num_regions * sizeof(float), 0,
                                       gpuMemcpyHostToDevice, stream));
    GPU_CHECK(gpuMemcpyToSymbolAsync(d_const_num_regions, &num_regions,
                                       sizeof(int), 0, gpuMemcpyHostToDevice, stream));
    GPU_CHECK(gpuMemcpyToSymbolAsync(d_const_num_spheres, &num_spheres,
                                       sizeof(int), 0, gpuMemcpyHostToDevice, stream));
    GPU_CHECK(gpuMemcpyToSymbolAsync(d_const_sphere_indices, sphere_indices,
                                       num_spheres * sizeof(int), 0,
                                       gpuMemcpyHostToDevice, stream));
}

// ============================================================================
// 查找材料衰减系数
// ============================================================================
float findMuForMaterial(const Material *materials, int num_materials,
                        const char *name, float energy) {
    float mu = interpolateMu(materials, num_materials, name, energy);
    if (mu > 0) return mu;

    fprintf(stderr, "Error: Could not find material '%s' for energy %.1f keV\n", name, energy);
    fprintf(stderr, "       Please check materials.csv and geometry configuration\n");
    return -1.0f;
}

// ============================================================================
// Main 函数
// ============================================================================
int main(int argc, char **argv)
{
    std::cout << "========================================" << std::endl;
    std::cout << "医学成像光子传输模拟 - Iluvatar CoreX" << std::endl;
    std::cout << "作者: weiwei2027" << std::endl;
    std::cout << "========================================" << std::endl;

    // 默认参数
    const char *geometry_file = "data/geometry_3layer.txt";
    const char *material_file = "data/materials.csv";
    const char *source_file = "data/source_point_10m.txt";
    const char *output_dir = "output/";
    double cpu_baseline = 0.0;  // CPU基线时间，0表示未提供

    // 解析命令行
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-g") == 0 && i + 1 < argc) geometry_file = argv[++i];
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) material_file = argv[++i];
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) source_file = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) output_dir = argv[++i];
        else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) cpu_baseline = atof(argv[++i]);
    }

    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Geometry: " << geometry_file << std::endl;
    std::cout << "  Materials: " << material_file << std::endl;
    std::cout << "  Source: " << source_file << std::endl;
    std::cout << "  Output: " << output_dir << std::endl;
    if (cpu_baseline > 0) {
        std::cout << "  CPU Baseline: " << cpu_baseline << " seconds" << std::endl;
    }

    // 读取输入文件
    Region *h_regions = nullptr;
    int num_regions = 0;
    if (parseGeometryFile(geometry_file, &h_regions, &num_regions) != 0) {
        std::cerr << "Error: Failed to parse geometry file" << std::endl;
        return 1;
    }

    Material *h_materials = nullptr;
    int num_materials = 0;
    if (parseMaterialFile(material_file, &h_materials, &num_materials) != 0) {
        std::cerr << "Error: Failed to parse material file" << std::endl;
        free(h_regions);
        return 1;
    }

    Source h_source;
    Detector h_detector;
    if (parseSourceFile(source_file, &h_source, &h_detector) != 0) {
        std::cerr << "Error: Failed to parse source file" << std::endl;
        free(h_regions);
        free(h_materials);
        return 1;
    }

    // 准备衰减系数
    float *h_mu_values = (float *)malloc(num_regions * sizeof(float));
    int num_layers = 0, num_spheres = 0;
    for (int i = 0; i < num_regions; i++) {
        h_mu_values[i] = findMuForMaterial(h_materials, num_materials,
                                           h_regions[i].material_name, h_source.energy);
        if (h_mu_values[i] < 0) {
            free(h_regions); free(h_materials); free(h_mu_values);
            return 1;
        }
        if (h_regions[i].type == LAYER) num_layers++;
        else num_spheres++;
    }
    std::cout << "Loaded: " << num_layers << " layers, " << num_spheres << " spheres" << std::endl;

    // 分配内存
    size_t detector_pixels = h_detector.nx * h_detector.ny;
    size_t detector_bytes = detector_pixels * sizeof(float);
    float *h_pixels = (float *)calloc(detector_pixels, sizeof(float));

    float *d_pixels = nullptr;
    gpuRandState *d_rand_states = nullptr;
    GPU_CHECK(gpuMalloc(&d_pixels, detector_bytes));
    GPU_CHECK(gpuMalloc(&d_rand_states, 65536 * sizeof(gpuRandState)));
    GPU_CHECK(gpuMemset(d_pixels, 0, detector_bytes));

    h_detector.pixels = d_pixels;

    // 拷贝几何数据到常量内存
    copyGeometryToConstant(h_regions, h_mu_values, num_regions);

    // 准备参数
    SimParams params = {num_regions, num_materials, h_source.num_photons, h_detector.z};


    // 配置 CUDA
    dim3 blockDim(256);
    dim3 gridDim(256);
    int total_threads = blockDim.x * gridDim.x;

    std::cout << "\nSimulation: " << h_source.num_photons << " photons, "
              << total_threads << " threads" << std::endl;

    // 清空探测器并初始化随机数
    clearDetectorKernel<<<gridDim, blockDim>>>(d_pixels, detector_pixels);
    initRandState<<<gridDim, blockDim>>>(d_rand_states, total_threads, 12345ULL);
    GPU_CHECK(gpuDeviceSynchronize());

    // 运行模拟
    auto start = std::chrono::high_resolution_clock::now();
    photonTransportKernel<<<gridDim, blockDim>>>(h_detector, params, d_rand_states, h_source);
    GPU_CHECK(gpuDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    // 拷贝结果
    GPU_CHECK(gpuMemcpy(h_pixels, d_pixels, detector_bytes, gpuMemcpyDeviceToHost));

    // 统计结果
    float total_detected = 0.0f, max_count = 0.0f;
    for (size_t i = 0; i < detector_pixels; i++) {
        total_detected += h_pixels[i];
        if (h_pixels[i] > max_count) max_count = h_pixels[i];
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "  Detected: " << (long long)total_detected << " / " << h_source.num_photons << std::endl;
    std::cout << "  Rate: " << (total_detected / h_source.num_photons * 100.0) << "%" << std::endl;
    std::cout << "  Time: " << elapsed.count() << " seconds" << std::endl;
    std::cout << "  Speed: " << (h_source.num_photons / elapsed.count()) << " photons/sec" << std::endl;

    // 保存结果
    char image_path[256], info_path[256], log_path[256];
    snprintf(image_path, sizeof(image_path), "%s/image.bin", output_dir);
    snprintf(info_path, sizeof(info_path), "%s/image_info.txt", output_dir);
    snprintf(log_path, sizeof(log_path), "%s/performance.log", output_dir);

    saveDetectorImage(image_path, h_pixels, h_detector.nx, h_detector.ny);
    saveImageInfo(info_path, h_detector.nx, h_detector.ny, "float32");
    savePerformanceLog(log_path, elapsed.count(), h_source.num_photons, cpu_baseline);

    std::cout << "\nOutput: " << output_dir << std::endl;

    // 清理
    free(h_regions); free(h_materials); free(h_mu_values); free(h_pixels);
    gpuFree(d_pixels); gpuFree(d_rand_states);

    return 0;
}
