/**
 * @file utils.h
 * @brief 工具函数接口 - 文件解析、几何计算、IO操作
 * @author weiwei2027
 * @date 2026-02-28
 */

#ifndef UTILS_H
#define UTILS_H

#include "types.h"

// ============================================================================
// 几何解析
// ============================================================================

/**
 * @brief 解析几何模型文件
 * @param filename 几何模型文件路径
 * @param regions 输出：区域数组（需调用者释放）
 * @param num_regions 输出：区域数量
 * @return 成功返回0，失败返回-1
 */
int parseGeometryFile(const char *filename, Region **regions, int *num_regions);

/**
 * @brief 打印几何模型信息（用于调试）
 */
void printGeometryInfo(const Region *regions, int num_regions);

/**
 * @brief 计算光子在当前区域的 z 方向边界距离
 * @param p 光子
 * @param region 当前区域
 * @return 到边界的距离，如果方向背离返回 -1
 */
float computeLayerBoundaryDistance(const Photon *p, const Region *region);

/**
 * @brief 计算光子到球体的相交距离
 * @param p 光子
 * @param region 球体区域
 * @param t_enter 输出：进入距离
 * @param t_exit 输出：离开距离
 * @return 是否相交
 */
bool computeSphereIntersection(const Photon *p, const Region *region,
                                float *t_enter, float *t_exit);

// ============================================================================
// 材料解析
// ============================================================================

/**
 * @brief 解析材料参数 CSV 文件
 * @param filename 材料参数文件路径
 * @param materials 输出：材料数组（需调用者释放）
 * @param num_materials 输出：材料数量
 * @return 成功返回0，失败返回-1
 */
int parseMaterialFile(const char *filename, Material **materials, int *num_materials);

/**
 * @brief 打印材料信息
 */
void printMaterialInfo(const Material *materials, int num_materials);

/**
 * @brief 根据材料名称查找材料ID
 * @param materials 材料数组
 * @param num_materials 材料数量
 * @param name 材料名称
 * @param energy 能量（用于匹配）
 * @return 材料ID，未找到返回-1
 */
int findMaterialId(const Material *materials, int num_materials,
                   const char *name, float energy);

/**
 * @brief 线性插值获取指定能量的衰减系数
 * @param materials 材料数组
 * @param num_materials 材料数量
 * @param name 材料名称
 * @param energy 目标能量
 * @return 插值后的 mu 值，失败返回 -1
 */
float interpolateMu(const Material *materials, int num_materials,
                    const char *name, float energy);

// ============================================================================
// 源参数解析
// ============================================================================

/**
 * @brief 解析源参数文件
 * @param filename 源参数文件路径
 * @param source 输出：源参数结构体
 * @param detector 输出：探测器参数结构体
 * @return 成功返回0，失败返回-1
 */
int parseSourceFile(const char *filename, Source *source, Detector *detector);

// ============================================================================
// IO 操作
// ============================================================================

/**
 * @brief 保存探测器图像到二进制文件
 * @param filename 输出文件路径
 * @param pixels 像素数据
 * @param nx 宽度（像素数）
 * @param ny 高度（像素数）
 * @return 成功返回0，失败返回-1
 */
int saveDetectorImage(const char *filename, const float *pixels, int nx, int ny);

/**
 * @brief 保存图像信息描述文件
 * @param filename 描述文件路径
 * @param nx 宽度
 * @param ny 高度
 * @param dtype 数据类型描述
 * @return 成功返回0，失败返回-1
 */
int saveImageInfo(const char *filename, int nx, int ny, const char *dtype);

/**
 * @brief 保存性能日志
 * @param filename 日志文件路径
 * @param sim_time 模拟时间（秒）
 * @param num_photons 光子数量
 * @param cpu_baseline CPU基准时间（用于计算加速比）
 * @return 成功返回0，失败返回-1
 */
int savePerformanceLog(const char *filename, double sim_time, int num_photons,
                       double cpu_baseline);

// ============================================================================
// 时间函数
// ============================================================================

/**
 * @brief 获取当前时间（秒）
 * @return 当前时间（从某个固定点开始的秒数）
 */
double getCurrentTime();

#endif // UTILS_H
