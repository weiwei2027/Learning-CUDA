/**
 * @file utils.cpp
 * @brief 工具函数实现 - 文件解析、几何计算、IO操作
 * @author weiwei2027
 * @date 2026-02-28
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include "../include/utils.h"

// ============================================================================
// 内部辅助函数
// ============================================================================

/**
 * @brief 手动解析浮点数（避免 sscanf 的 glibc 兼容性问题）
 */
static float parseFloat(const std::string& str)
{
    float result = 0.0f;
    bool afterDot = false;
    float fraction = 0.1f;
    bool negative = false;
    
    for (char c : str) {
        if (c == '-') {
            negative = true;
        } else if (c >= '0' && c <= '9') {
            if (!afterDot) {
                result = result * 10.0f + (c - '0');
            } else {
                result += (c - '0') * fraction;
                fraction *= 0.1f;
            }
        } else if (c == '.') {
            afterDot = true;
        }
    }
    
    return negative ? -result : result;
}

/**
 * @brief 去除字符串首尾空格
 */
static std::string trim(const std::string& str)
{
    size_t start = str.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}

// ============================================================================
// 几何解析
// ============================================================================

/**
 * @brief 解析几何模型文件
 * 
 * 文件格式: type name thickness x y z material
 * 示例: layer skin 0.2 0.0 0.0 0.0 tissue_skin
 * 支持空行和 # 开头的注释行
 */
int parseGeometryFile(const char *filename, Region **regions, int *num_regions)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open geometry file: %s\n", filename);
        return -1;
    }
    
    // 先统计有效行数（非空、非注释）
    int count = 0;
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] != '\n' && line[0] != '#' && line[0] != '\r') {
            count++;
        }
    }
    
    if (count == 0) {
        *regions = nullptr;
        *num_regions = 0;
        fclose(fp);
        return 0;
    }
    
    // 分配内存
    *regions = (Region *)malloc(count * sizeof(Region));
    if (!*regions) {
        fprintf(stderr, "Error: Failed to allocate memory for regions\n");
        fclose(fp);
        return -1;
    }
    
    // 解析文件内容
    rewind(fp);
    int idx = 0;
    
    while (fgets(line, sizeof(line), fp) && idx < count) {
        // 跳过空行和注释行
        if (line[0] == '\n' || line[0] == '#' || line[0] == '\r') continue;
        
        // 去除行尾换行符
        line[strcspn(line, "\r\n")] = '\0';
        
        // 解析行: type name thickness x y z material
        std::istringstream iss(line);
        std::string type_str, name, material;
        float thickness, x, y, z;
        
        if (iss >> type_str >> name >> thickness >> x >> y >> z >> material) {
            // 设置区域类型
            if (type_str == "layer") {
                (*regions)[idx].type = LAYER;
            } else if (type_str == "sphere") {
                (*regions)[idx].type = SPHERE;
            } else {
                fprintf(stderr, "Warning: Unknown region type: %s\n", type_str.c_str());
                continue;
            }
            
            // 复制字符串
            strncpy((*regions)[idx].name, name.c_str(), 31);
            (*regions)[idx].name[31] = '\0';
            strncpy((*regions)[idx].material_name, material.c_str(), 31);
            (*regions)[idx].material_name[31] = '\0';
            
            // 设置数值
            (*regions)[idx].thickness = thickness;
            (*regions)[idx].x = x;
            (*regions)[idx].y = y;
            (*regions)[idx].z = z;
            (*regions)[idx].material_id = -1;  // 待关联
            
            idx++;
        } else {
            fprintf(stderr, "Warning: Failed to parse line: %s\n", line);
        }
    }
    
    *num_regions = idx;
    fclose(fp);
    
    printf("Loaded %d regions from %s\n", *num_regions, filename);
    return 0;
}

void printGeometryInfo(const Region *regions, int num_regions)
{
    printf("\n=== Geometry Information ===\n");
    printf("Total regions: %d\n\n", num_regions);
    
    for (int i = 0; i < num_regions; i++) {
        printf("Region %d:\n", i);
        printf("  Type: %s\n", regions[i].type == LAYER ? "Layer" : "Sphere");
        printf("  Name: %s\n", regions[i].name);
        printf("  Thickness/Radius: %.2f cm\n", regions[i].thickness);
        printf("  Position: (%.2f, %.2f, %.2f)\n", 
               regions[i].x, regions[i].y, regions[i].z);
        printf("  Material ID: %d\n", regions[i].material_id);
        printf("\n");
    }
}

/**
 * @brief 计算光子到平板层边界的距离
 * 
 * 平板层沿 z 轴堆叠，计算光子沿当前方向到达层边界的距离。
 * 距离公式: t = (边界 - 当前位置) / dz
 * 
 * @return 到边界的距离，方向背离或平行返回 -1
 */
float computeLayerBoundaryDistance(const Photon *p, const Region *region)
{
    // 平板层假设沿 z 轴堆叠
    float z_boundary = region->z + region->thickness;
    
    if (p->dz > 0) {
        // 光子向前移动
        float t = (z_boundary - p->z) / p->dz;
        return t > 0 ? t : -1.0f;
    } else if (p->dz < 0) {
        // 光子向后移动
        float t = (region->z - p->z) / p->dz;
        return t > 0 ? t : -1.0f;
    }
    
    return -1.0f;  // 平行于层
}

/**
 * @brief 计算光线与球体的相交距离
 * 
 * 解二次方程求光线与球面的交点。
 * 方程: t^2 + 2(O·D)t + (|O|^2 - R^2) = 0, 其中 O = P - C
 * 
 * @param t_enter 输出：进入球体的距离
 * @param t_exit  输出：离开球体的距离
 * @return 是否相交
 */
bool computeSphereIntersection(const Photon *p, const Region *region,
                                float *t_enter, float *t_exit)
{
    // 射线-球相交公式
    float ox = p->x - region->x;
    float oy = p->y - region->y;
    float oz = p->z - region->z;
    
    float b = 2.0f * (ox * p->dx + oy * p->dy + oz * p->dz);
    float c = ox * ox + oy * oy + oz * oz - region->thickness * region->thickness;
    
    float discriminant = b * b - 4.0f * c;
    
    if (discriminant < 0) return false;
    
    float sqrt_disc = sqrtf(discriminant);
    *t_enter = (-b - sqrt_disc) / 2.0f;
    *t_exit = (-b + sqrt_disc) / 2.0f;
    
    return true;
}

// ============================================================================
// 材料解析
// ============================================================================

int parseMaterialFile(const char *filename, Material **materials, int *num_materials)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open material file: %s\n", filename);
        return -1;
    }
    
    // 跳过表头
    char line[256];
    if (!fgets(line, sizeof(line), fp)) {
        fprintf(stderr, "Error: Empty material file\n");
        fclose(fp);
        return -1;
    }
    
    // 统计数据行数
    int count = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] != '\n' && line[0] != '\r') count++;
    }
    
    if (count == 0) {
        *materials = nullptr;
        *num_materials = 0;
        fclose(fp);
        return 0;
    }
    
    // 分配内存
    *materials = (Material *)malloc(count * sizeof(Material));
    if (!*materials) {
        fprintf(stderr, "Error: Failed to allocate memory for materials\n");
        fclose(fp);
        return -1;
    }
    
    // 解析 CSV
    rewind(fp);
    fgets(line, sizeof(line), fp);  // 跳过表头
    
    int idx = 0;
    while (fgets(line, sizeof(line), fp) && idx < count) {
        if (line[0] == '\n' || line[0] == '\r') continue;
        
        // 去除行尾换行符
        line[strcspn(line, "\r\n")] = '\0';
        
        // 手动解析 CSV: material,energy,mu
        std::string str(line);
        size_t comma1 = str.find(',');
        if (comma1 == std::string::npos) continue;
        
        size_t comma2 = str.find(',', comma1 + 1);
        if (comma2 == std::string::npos) continue;
        
        std::string name = trim(str.substr(0, comma1));
        std::string energy_str = trim(str.substr(comma1 + 1, comma2 - comma1 - 1));
        std::string mu_str = trim(str.substr(comma2 + 1));
        
        strncpy((*materials)[idx].name, name.c_str(), 31);
        (*materials)[idx].name[31] = '\0';
        (*materials)[idx].energy = parseFloat(energy_str);
        (*materials)[idx].mu = parseFloat(mu_str);
        idx++;
    }
    
    *num_materials = idx;
    fclose(fp);
    
    printf("Loaded %d material entries from %s\n", *num_materials, filename);
    return 0;
}

void printMaterialInfo(const Material *materials, int num_materials)
{
    printf("\n=== Material Information ===\n");
    printf("Total entries: %d\n\n", num_materials);
    
    for (int i = 0; i < num_materials; i++) {
        printf("Material %d:\n", i);
        printf("  Name: %s\n", materials[i].name);
        printf("  Energy: %.1f keV\n", materials[i].energy);
        printf("  mu: %.4f cm^-1\n", materials[i].mu);
        printf("\n");
    }
}

int findMaterialId(const Material *materials, int num_materials,
                   const char *name, float energy)
{
    for (int i = 0; i < num_materials; i++) {
        if (strcmp(materials[i].name, name) == 0) {
            // 如果指定了能量，检查是否匹配（允许小误差）
            if (energy > 0 && fabsf(materials[i].energy - energy) > 0.1f) {
                continue;
            }
            return i;
        }
    }
    return -1;
}

/**
 * @brief 线性插值获取指定能量的衰减系数
 * 
 * 在材料数据表中找到目标能量的上下界，进行线性插值。
 * 公式: mu = mu_low + (energy - e_low) / (e_high - e_low) * (mu_high - mu_low)
 */
float interpolateMu(const Material *materials, int num_materials,
                    const char *name, float energy)
{
    // 找到该材料的所有能量点
    int lower_idx = -1, upper_idx = -1;
    float lower_energy = 0.0f, upper_energy = 1e30f;
    
    for (int i = 0; i < num_materials; i++) {
        if (strcmp(materials[i].name, name) == 0) {
            float e = materials[i].energy;
            if (e <= energy && e > lower_energy) {
                lower_energy = e;
                lower_idx = i;
            }
            if (e >= energy && e < upper_energy) {
                upper_energy = e;
                upper_idx = i;
            }
        }
    }
    
    if (lower_idx == -1 && upper_idx == -1) {
        return -1.0f;  // 未找到该材料
    }
    if (lower_idx == -1) return materials[upper_idx].mu;
    if (upper_idx == -1) return materials[lower_idx].mu;
    if (lower_idx == upper_idx) return materials[lower_idx].mu;
    
    // 线性插值
    float t = (energy - lower_energy) / (upper_energy - lower_energy);
    return materials[lower_idx].mu + t * (materials[upper_idx].mu - materials[lower_idx].mu);
}

// ============================================================================
// 源参数解析
// ============================================================================

int parseSourceFile(const char *filename, Source *source, Detector *detector)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Cannot open source file: %s\n", filename);
        return -1;
    }
    
    // 默认值
    memset(source, 0, sizeof(Source));
    memset(detector, 0, sizeof(Detector));
    source->type = 0;  // 默认为点源
    
    std::string line;
    while (std::getline(file, line)) {
        // 跳过空行和注释
        if (line.empty() || line[0] == '#') continue;
        
        // 解析键值对 (key = value)
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;
        
        std::string key = trim(line.substr(0, eq_pos));
        std::string value = trim(line.substr(eq_pos + 1));
        
        if (key == "source_type") {
            if (value == "point") source->type = 0;
            else if (value == "parallel") source->type = 1;
        }
        else if (key == "position") {
            std::istringstream iss(value);
            iss >> source->x >> source->y >> source->z;
        }
        else if (key == "direction") {
            std::istringstream iss(value);
            iss >> source->dx >> source->dy >> source->dz;
        }
        else if (key == "energy") {
            source->energy = parseFloat(value);
        }
        else if (key == "num_photons") {
            source->num_photons = (int)parseFloat(value);
        }
        else if (key == "detector_position") {
            std::istringstream iss(value);
            iss >> detector->x >> detector->y >> detector->z;
        }
        else if (key == "detector_size") {
            std::istringstream iss(value);
            iss >> detector->width >> detector->height;
        }
        else if (key == "detector_pixels") {
            std::istringstream iss(value);
            iss >> detector->nx >> detector->ny;
        }
    }
    
    file.close();
    
    printf("Loaded source configuration from %s\n", filename);
    printf("  Source: %s at (%.2f, %.2f, %.2f)\n", 
           source->type == 0 ? "Point" : "Parallel",
           source->x, source->y, source->z);
    printf("  Photons: %d, Energy: %.1f keV\n", source->num_photons, source->energy);
    printf("  Detector: %dx%d pixels at z=%.2f\n", 
           detector->nx, detector->ny, detector->z);
    
    return 0;
}

// ============================================================================
// IO 操作
// ============================================================================

int saveDetectorImage(const char *filename, const float *pixels, int nx, int ny)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot create output file: %s\n", filename);
        return -1;
    }
    
    // 写入元数据头
    int header[2] = {nx, ny};
    fwrite(header, sizeof(int), 2, fp);
    
    // 写入像素数据
    size_t num_pixels = nx * ny;
    size_t written = fwrite(pixels, sizeof(float), num_pixels, fp);
    
    fclose(fp);
    
    if (written != num_pixels) {
        fprintf(stderr, "Error: Failed to write all pixels\n");
        return -1;
    }
    
    printf("Saved detector image to %s (%d x %d pixels)\n", filename, nx, ny);
    return 0;
}

int saveImageInfo(const char *filename, int nx, int ny, const char *dtype)
{
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot create info file: %s\n", filename);
        return -1;
    }
    
    fprintf(fp, "width=%d\n", nx);
    fprintf(fp, "height=%d\n", ny);
    fprintf(fp, "dtype=%s\n", dtype);
    fprintf(fp, "format=binary\n");
    
    fclose(fp);
    return 0;
}

int savePerformanceLog(const char *filename, double sim_time, int num_photons,
                        double cpu_baseline)
{
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot create log file: %s\n", filename);
        return -1;
    }
    
    double photon_rate = num_photons / sim_time;
    bool has_cpu_baseline = (cpu_baseline > 0);
    
    fprintf(fp, "=== Photon Transport Simulation Performance Log ===\n");
    fprintf(fp, "Simulation Time: %.4f seconds\n", sim_time);
    fprintf(fp, "Total Photons: %d\n", num_photons);
    fprintf(fp, "Photon Processing Rate: %.2e photons/sec\n", photon_rate);
    if (has_cpu_baseline) {
        double speedup = cpu_baseline / sim_time;
        fprintf(fp, "Speedup vs CPU: %.2fx\n", speedup);
    }
    
    fclose(fp);
    
    printf("Saved performance log to %s\n", filename);
    printf("  Processing rate: %.2e photons/sec\n", photon_rate);
    if (has_cpu_baseline) {
        double speedup = cpu_baseline / sim_time;
        printf("  Speedup: %.2fx\n", speedup);
    }
    
    return 0;
}

// ============================================================================
// 时间函数
// ============================================================================

double getCurrentTime()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
