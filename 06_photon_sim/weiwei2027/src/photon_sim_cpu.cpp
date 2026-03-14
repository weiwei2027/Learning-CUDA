/**
 * CPU 基准版本 - 光子传输模拟
 * 单线程实现，用于与 GPU 版本对比加速比
 */

#include <iostream>
#include <fstream>
#include <chrono>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <sys/stat.h>
#include "../include/types.h"


#include "../include/utils.h"

/**
 * @brief 查找指定材料在给定能量下的衰减系数
 * @param materials 材料数组
 * @param num_materials 材料数量
 * @param name 材料名称
 * @param energy 光子能量
 * @return 衰减系数 mu (cm^-1)
 */
float findMuForMaterial(const Material* materials, int num_materials, const char* name, float energy)
{
    // 使用线性插值查找材料衰减系数（支持50-100keV范围）
    // 返回-1.0f表示未找到材料（错误）
    float mu = interpolateMu(materials, num_materials, name, energy);
    if (mu > 0) return mu;

    // 如果找不到，返回错误标记
    fprintf(stderr, "Error: Could not find material '%s' for energy %.1f keV\n", name, energy);
    fprintf(stderr, "       Please check materials.csv and geometry.txt configuration\n");
    return -1.0f;
}


/**
 * @brief CPU 版本的自由程采样
 * @param mu 线性衰减系数
 * @param gen 随机数生成器
 * @return 自由程距离
 */
float sampleFreePathCPU(float mu, std::mt19937& gen)
{
    if (mu <= 0.0f) return 1e30f;

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float xi = dist(gen);
    if (xi < 1e-7f) xi = 1e-7f;
    return -logf(xi) / mu;
}

/**
 * @brief CPU 版本的光子初始化
 */
void initPhotonCPU(Photon* p, float src_x, float src_y, float src_z, float energy)
{
    p->x = src_x;
    p->y = src_y;
    p->z = src_z;
    p->dx = 0.0f;
    p->dy = 0.0f;
    p->dz = 1.0f;
    p->energy = energy;
    p->alive = true;
}

/**
 * @brief 光线-球体相交检测
 * @param px, py, pz 光子位置
 * @param dx, dy, dz 光子方向（单位向量）
 * @param cx, cy, cz 球心
 * @param radius 球半径
 * @param t_enter 输出：进入距离
 * @param t_exit 输出：离开距离
 * @return bool 是否相交
 */
bool intersectSphereCPU(
    float px, float py, float pz,
    float dx, float dy, float dz,
    float cx, float cy, float cz, float radius,
    float& t_enter, float& t_exit)
{
    // 计算球心到光子起点的向量
    float ox = cx - px;
    float oy = cy - py;
    float oz = cz - pz;

    // 由于方向向量已经归一化，a = 1.0
    // b = -2 * (D · O)
    float b = -2.0f * (dx * ox + dy * oy + dz * oz);

    // c = |O|² - R²
    float c = ox * ox + oy * oy + oz * oz - radius * radius;

    // 判别式 Δ = b² - 4c
    float discriminant = b * b - 4.0f * c;

    if (discriminant < 0.0f)
    {
        return false; // 不相交
    }

    float sqrt_disc = sqrtf(discriminant);

    // 求解 t
    t_enter = (-b - sqrt_disc) * 0.5f;
    t_exit = (-b + sqrt_disc) * 0.5f;

    // 如果 t_exit < 0，球体在光子后方
    if (t_exit < 0.0f)
    {
        return false;
    }

    // 确保相交区间有效
    if (t_enter < 0.0f)
    {
        t_enter = 0.0f; // 从起点就在球内
    }

    return true;
}

/**
 * @brief 计算球体区域的光学路径长度
 * @param p 光子
 * @param region 球体区域
 * @param detector_z 探测器z坐标
 * @param t_enter 输出：进入距离
 * @param t_exit 输出：离开距离
 * @return bool 是否穿过球体
 */
bool computeSpherePathCPU(
    const Photon& p,
    const Region& region,
    float detector_z,
    float& t_enter, float& t_exit)
{
    if (region.type != SPHERE) return false;

    return intersectSphereCPU(
        p.x, p.y, p.z, p.dx, p.dy, p.dz,
        region.x, region.y, region.z, region.thickness, // thickness 是半径
        t_enter, t_exit
    );
}

/**
 * @brief CPU 单线程光子传输模拟（支持平板层和球体）
 * @param regions 几何区域数组
 * @param materials_mu 材料衰减系数数组
 * @param num_regions 区域数量
 * @param detector 探测器参数
 * @param params 模拟参数
 * @param source 源参数
 * @param seed 随机数种子
 */
void photonTransportCPU(
    const Region* regions,
    const float* materials_mu,
    int num_regions,
    Detector& detector,
    const SimParams& params,
    const Source& source,
    unsigned int seed)
{
    // 初始化随机数生成器（每个光子独立种子，确保可重复性）
    std::mt19937 gen(seed);
    
    // 计算探测器边界（用于点源和平行束的采样范围）
    float det_x_min = detector.x - detector.width / 2.0f;
    float det_x_max = detector.x + detector.width / 2.0f;
    float det_y_min = detector.y - detector.height / 2.0f;
    float det_y_max = detector.y + detector.height / 2.0f;
    
    // 点源模式：计算探测器边界相对于源点的角度范围
    float dz = detector.z - source.z;
    float angle_x_min = atanf((det_x_min - source.x) / dz);
    float angle_x_max = atanf((det_x_max - source.x) / dz);
    float angle_y_min = atanf((det_y_min - source.y) / dz);
    float angle_y_max = atanf((det_y_max - source.y) / dz);
    
    // 预创建随机数分布器
    std::uniform_real_distribution<float> dist_angle_x(angle_x_min, angle_x_max);
    std::uniform_real_distribution<float> dist_angle_y(angle_y_min, angle_y_max);
    std::uniform_real_distribution<float> dist_x(det_x_min, det_x_max);
    std::uniform_real_distribution<float> dist_y(det_y_min, det_y_max);

    int total_pixels = detector.nx * detector.ny;

    // 清零探测器
    for (int i = 0; i < total_pixels; i++)
    {
        detector.pixels[i] = 0.0f;
    }

    // 逐个光子模拟
    for (int photon_id = 0; photon_id < params.num_photons; photon_id++)
    {
        Photon p;
        
        // 根据源类型初始化光子
        if (source.type == 1)  // parallel mode
        {
            // 平行束：在源平面上均匀分布（覆盖探测器范围），方向固定
            // 位置在探测器投影范围内均匀采样，确保全部光子都能到达探测器
            float x = dist_x(gen);
            float y = dist_y(gen);
            initPhotonCPU(&p, x, y, source.z, source.energy);
            // 方向固定为源的方向（默认 0,0,1）
            p.dx = source.dx;
            p.dy = source.dy;
            p.dz = source.dz;
        }
        else  // point mode (default)
        {
            // 点源：在角度空间均匀采样，生成覆盖整个探测器的锥形束
            // 初始位置固定在源点
            initPhotonCPU(&p, source.x, source.y, source.z, source.energy);
            
            // 在探测器张角范围内均匀采样角度
            float angle_x = dist_angle_x(gen);  // 水平方向发散角
            float angle_y = dist_angle_y(gen);  // 垂直方向发散角
            
            // 从角度计算方向向量：tan(θ) = 对边/邻边
            float tan_x = tanf(angle_x);
            float tan_y = tanf(angle_y);
            
            // 归一化方向向量：len = sqrt(tan²x + tan²y + 1)
            float len = sqrtf(tan_x * tan_x + tan_y * tan_y + 1.0f);
            p.dx = tan_x / len;
            p.dy = tan_y / len;
            p.dz = 1.0f / len;
        }

        float current_z = p.z;
        bool absorbed = false;

        // 传播到第一层入口（处理源点到几何模型起点的距离）
        if (num_regions > 0) {
            float first_layer_start = 1e30f;
            for (int i = 0; i < num_regions; i++) {
                if (regions[i].type == LAYER) {
                    first_layer_start = fminf(first_layer_start, regions[i].z);
                }
            }
            if (current_z < first_layer_start && p.dz > 0) {
                float dz_to_first = first_layer_start - current_z;
                float path_to_first = dz_to_first / p.dz;
                p.x += p.dx * path_to_first;
                p.y += p.dy * path_to_first;
                current_z = first_layer_start;
            }
        }

        // 逐层传输，同时处理球体
        for (int r = 0; r < num_regions; r++)
        {
            if (regions[r].type != LAYER) continue;

            float layer_start = regions[r].z;
            float layer_end = regions[r].z + regions[r].thickness;

            if (current_z >= layer_end) continue;

            float enter_z = fmaxf(current_z, layer_start);
            float exit_z = fminf(params.detector_z, layer_end);

            if (exit_z <= enter_z) continue;

            // 计算该层的路径参数
            float dz_layer = exit_z - enter_z;
            float layer_path_length = dz_layer / fabsf(p.dz + 1e-10f); // 实际物理路径

            // 检查该层内是否有球体
            float remaining_path = layer_path_length;
            float current_dist = 0.0f; // 在当前层内已走的距离
            bool hit_sphere = false;

            // 计算层入口处的实际 x, y 位置
            float z_offset = enter_z - p.z;
            float layer_enter_x = p.x + p.dx * z_offset / (p.dz + 1e-10f);
            float layer_enter_y = p.y + p.dy * z_offset / (p.dz + 1e-10f);

            for (int s = 0; s < num_regions && !hit_sphere; s++)
            {
                if (regions[s].type != SPHERE) continue;

                // 计算光线与球体的相交
                float t_enter, t_exit;
                // 使用层入口处的位置作为起点

                if (intersectSphereCPU(
                    layer_enter_x, layer_enter_y, enter_z, // 起点：层入口
                    p.dx, p.dy, p.dz, // 方向
                    regions[s].x, regions[s].y, regions[s].z, regions[s].thickness, // 球心和半径
                    t_enter, t_exit))
                {
                    // 检查相交是否在层的范围内
                    if (t_enter < layer_path_length && t_exit > 0)
                    {
                        hit_sphere = true;

                        // 分段处理路径
                        // 段1: 层起点到球体入口
                        float d1 = fminf(t_enter, layer_path_length);
                        if (d1 > 0)
                        {
                            float fp1 = sampleFreePathCPU(materials_mu[r], gen);
                            if (fp1 < d1)
                            {
                                absorbed = true;
                                break;
                            }
                        }

                        if (absorbed) break;

                        // 段2: 球体内部
                        float d2_start = fmaxf(t_enter, 0.0f);
                        float d2_end = fminf(t_exit, layer_path_length);
                        float d2 = d2_end - d2_start;

                        if (d2 > 0)
                        {
                            float fp2 = sampleFreePathCPU(materials_mu[s], gen);
                            if (fp2 < d2)
                            {
                                absorbed = true;
                                break;
                            }
                        }

                        if (absorbed) break;

                        // 段3: 球体出口到层终点
                        float d3 = layer_path_length - fminf(t_exit, layer_path_length);
                        if (d3 > 0)
                        {
                            float fp3 = sampleFreePathCPU(materials_mu[r], gen);
                            if (fp3 < d3)
                            {
                                absorbed = true;
                                break;
                            }
                        }
                    }
                }
            }

            if (absorbed) break;

            // 如果没有遇到球体或球体不在路径上，按原方式处理
            if (!hit_sphere)
            {
                float free_path = sampleFreePathCPU(materials_mu[r], gen);
                if (free_path < layer_path_length)
                {
                    absorbed = true;
                    break;
                }
            }

            // 更新位置（沿直线传播，使用实际物理路径长度）
            p.x += p.dx * layer_path_length;
            p.y += p.dy * layer_path_length;
            current_z = exit_z;
        }

        // 传播到探测器平面（处理最后一层到探测器的距离）
        if (!absorbed && current_z < detector.z)
        {
            float dz_to_detector = detector.z - current_z;
            float path_to_detector = dz_to_detector / fabsf(p.dz + 1e-10f);
            p.x += p.dx * path_to_detector;
            p.y += p.dy * path_to_detector;
        }

        if (!absorbed)
        {
            // 映射到探测器像素
            int px = (int)((p.x + detector.width / 2.0f) / detector.width * detector.nx);
            int py = (int)((p.y + detector.height / 2.0f) / detector.height * detector.ny);

            // 边界检查
            px = std::max(0, std::min(px, detector.nx - 1));
            py = std::max(0, std::min(py, detector.ny - 1));

            int idx = py * detector.nx + px;
            detector.pixels[idx] += 1.0f;
        }
    }
}

/**
 * @brief 计算总光子数
 */
float sumDetectorPixels(const float* pixels, int nx, int ny)
{
    float sum = 0.0f;
    for (int i = 0; i < nx * ny; i++)
    {
        sum += pixels[i];
    }
    return sum;
}

void printUsage(const char* program_name)
{
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -g <file>   Geometry file (default: data/geometry.txt)" << std::endl;
    std::cout << "  -m <file>   Materials file (default: data/materials.csv)" << std::endl;
    std::cout << "  -s <file>   Source file (default: data/source.txt)" << std::endl;
    std::cout << "  -o <dir>    Output directory (default: output/)" << std::endl;
    std::cout << "  -h          Show this help" << std::endl;
}

int main(int argc, char** argv)
{
    std::cout << "========================================" << std::endl;
    std::cout << "医学成像光子传输模拟 - CPU 基准版本" << std::endl;
    std::cout << "作者: weiwei2027" << std::endl;
    std::cout << "========================================" << std::endl;

    // 默认文件路径
    const char* geometry_file = "data/geometry.txt";
    const char* material_file = "data/materials.csv";
    const char* source_file = "data/source.txt";
    const char* output_dir = "output/";

    // 解析命令行参数
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            printUsage(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "-g") == 0 && i + 1 < argc)
        {
            geometry_file = argv[++i];
        }
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc)
        {
            material_file = argv[++i];
        }
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc)
        {
            source_file = argv[++i];
        }
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc)
        {
            output_dir = argv[++i];
        }
    }

    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Geometry: " << geometry_file << std::endl;
    std::cout << "  Materials: " << material_file << std::endl;
    std::cout << "  Source: " << source_file << std::endl;
    std::cout << "  Output: " << output_dir << std::endl;

    // 读取几何模型文件
    Region* h_regions = nullptr;
    int num_regions = 0;
    if (parseGeometryFile(geometry_file, &h_regions, &num_regions) != 0)
    {
        std::cerr << "Error: Failed to parse geometry file" << std::endl;
        return 1;
    }

    // 读取物理参数文件
    Material* h_materials = nullptr;
    int num_materials = 0;
    if (parseMaterialFile(material_file, &h_materials, &num_materials) != 0)
    {
        std::cerr << "Error: Failed to parse material file" << std::endl;
        free(h_regions);
        return 1;
    }

    // 读取源参数文件
    Source h_source;
    Detector h_detector;
    if (parseSourceFile(source_file, &h_source, &h_detector) != 0)
    {
        std::cerr << "Error: Failed to parse source file" << std::endl;
        free(h_regions);
        free(h_materials);
        return 1;
    }

    // 为每个区域准备衰减系数（使用源的能量，支持50-100keV范围）
    float* h_mu_values = (float*)malloc(num_regions * sizeof(float));
    for (int i = 0; i < num_regions; i++)
    {
        h_mu_values[i] = findMuForMaterial(h_materials, num_materials, h_regions[i].material_name, h_source.energy);
        if (h_mu_values[i] < 0) {
            std::cerr << "Error: Failed to find material '" << h_regions[i].material_name << "' for region '" << h_regions[i].name << "'" << std::endl;
            free(h_regions);
            free(h_materials);
            free(h_mu_values);
            return 1;
        }
        const char* type_str = (h_regions[i].type == LAYER) ? "layer" : "sphere";
        if (h_regions[i].type == SPHERE)
        {
            std::cout << "Region " << h_regions[i].name << " (" << type_str << "): mu = "
                << h_mu_values[i] << " cm^-1, center=("
                << h_regions[i].x << ", " << h_regions[i].y << ", " << h_regions[i].z << "), r="
                << h_regions[i].thickness << std::endl;
        }
        else
        {
            std::cout << "Region " << h_regions[i].name << " (" << type_str << "): mu = "
                << h_mu_values[i] << " cm^-1" << std::endl;
        }
    }

    // 分配探测器像素数组
    int total_pixels = h_detector.nx * h_detector.ny;
    float* h_pixels = (float*)calloc(total_pixels, sizeof(float));
    h_detector.pixels = h_pixels;

    // 模拟参数
    SimParams params;
    params.num_regions = num_regions;
    params.num_materials = num_materials;
    params.num_photons = h_source.num_photons;
    params.detector_z = h_detector.z;

    std::cout << "\nSimulation Configuration:" << std::endl;
    std::cout << "  Photons: " << params.num_photons << std::endl;
    std::cout << "  Mode: CPU Single Thread" << std::endl;
    std::cout << "  Detector: " << h_detector.nx << "x" << h_detector.ny << std::endl;

    // 开始计时
    auto start = std::chrono::high_resolution_clock::now();

    // 执行光子传输模拟
    photonTransportCPU(h_regions, h_mu_values, num_regions, h_detector, params, h_source, 12345);

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // 统计结果
    float total_detected = sumDetectorPixels(h_pixels, h_detector.nx, h_detector.ny);
    float detection_rate = total_detected / params.num_photons * 100.0f;

    std::cout << "\nSimulation Complete:" << std::endl;
    std::cout << "  Detected photons: " << (int)total_detected << std::endl;
    std::cout << "  Detection rate: " << detection_rate << "%" << std::endl;
    std::cout << "  Time elapsed: " << elapsed.count() << " seconds" << std::endl;


    double photon_rate = params.num_photons / elapsed.count();
    std::cout << "  Photon processing rate: " << photon_rate << " photons/sec" << std::endl;

    // 自动创建输出目录（如果不存在）
    struct stat st = {0};
    if (stat(output_dir, &st) == -1) {
        mkdir(output_dir, 0755);
    }

    // 保存结果
    char output_filename[256];
    const char* sep = (strlen(output_dir) > 0 && output_dir[strlen(output_dir)-1] == '/') ? "" : "/";
    snprintf(output_filename, sizeof(output_filename), "%s%simage_cpu.bin", output_dir, sep);
    saveDetectorImage(output_filename, h_pixels, h_detector.nx, h_detector.ny);

    // 保存图像信息文件
    char info_filename[256];
    snprintf(info_filename, sizeof(info_filename), "%s%simage_cpu_info.txt", output_dir, sep);
    saveImageInfo(info_filename, h_detector.nx, h_detector.ny, "float32");

    // 保存性能日志
    char log_filename[256];
    snprintf(log_filename, sizeof(log_filename), "%s%sperformance_cpu.log", output_dir, sep);
    savePerformanceLog(log_filename, elapsed.count(), params.num_photons, elapsed.count()); // cpu_baseline=自身运行时间，speedup=1.0x

    // 清理内存
    free(h_regions);
    free(h_materials);
    free(h_mu_values);
    free(h_pixels);

    std::cout << "\nCPU Baseline simulation complete!" << std::endl;
    return 0;
}
