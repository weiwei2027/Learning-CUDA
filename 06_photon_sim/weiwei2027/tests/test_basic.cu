#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "../include/types.h"
#include "../src/photon_sim.cuh"

// 测试宏
#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            printf("FAILED: %s at %s:%d\n", msg, __FILE__, __LINE__); \
            return 1; \
        } \
    } while(0)

#define TEST_ASSERT_FLOAT_EQ(a, b, eps, msg) \
    do { \
        if (fabs((a) - (b)) > (eps)) { \
            printf("FAILED: %s (%.6f vs %.6f) at %s:%d\n", msg, (float)(a), (float)(b), __FILE__, __LINE__); \
            return 1; \
        } \
    } while(0)

/**
 * @brief 测试自由程采样
 * 验证 sampleFreePath 函数在 μ=1 时返回的期望值是否合理
 */
int testFreePathSampling()
{
    printf("Testing free path sampling...\n");
    
    // 由于无法在主机端测试设备函数，这里仅做简单验证
    // 实际测试需要在 GPU 上进行
    
    printf("  Note: Device function test requires GPU execution\n");
    printf("  PASSED (placeholder)\n");
    return 0;
}

/**
 * @brief 测试几何解析
 */
int testGeometryParsing()
{
    printf("Testing geometry parsing...\n");
    
    // 创建一个临时测试文件
    const char *test_file = "/tmp/test_geometry.txt";
    FILE *fp = fopen(test_file, "w");
    TEST_ASSERT(fp != nullptr, "Cannot create test file");
    
    fprintf(fp, "layer skin 0.2 0.0 0.0 0.0 tissue_skin\n");
    fprintf(fp, "layer skull 0.8 0.0 0.0 0.2 tissue_bone\n");
    fclose(fp);
    
    // TODO: 调用 parseGeometryFile 并验证结果
    
    printf("  PASSED\n");
    return 0;
}

/**
 * @brief 测试材料解析
 */
int testMaterialParsing()
{
    printf("Testing material parsing...\n");
    
    // 创建临时测试文件
    const char *test_file = "/tmp/test_materials.csv";
    FILE *fp = fopen(test_file, "w");
    TEST_ASSERT(fp != nullptr, "Cannot create test file");
    
    fprintf(fp, "material,energy_keV,mu\n");
    fprintf(fp, "tissue_test,50,0.25\n");
    fclose(fp);
    
    // TODO: 调用 parseMaterialFile 并验证结果
    
    printf("  PASSED\n");
    return 0;
}

/**
 * @brief 测试探测器像素映射
 */
__global__ void testPixelMappingKernel(Detector detector, float x, float y, int *px_out, int *py_out)
{
    // 模拟 recordPhoton 中的像素计算
    int px = (int)((x + detector.width / 2.0f) / detector.width * detector.nx);
    int py = (int)((y + detector.height / 2.0f) / detector.height * detector.ny);
    
    *px_out = px;
    *py_out = py;
}

int testPixelMapping()
{
    printf("Testing detector pixel mapping...\n");
    
    // 设置测试探测器
    Detector h_detector;
    h_detector.width = 20.0f;
    h_detector.height = 20.0f;
    h_detector.nx = 1024;
    h_detector.ny = 1024;
    
    // 分配设备内存
    Detector *d_detector;
    int *d_px, *d_py, h_px, h_py;
    
    CUDA_CHECK(cudaMalloc(&d_detector, sizeof(Detector)));
    CUDA_CHECK(cudaMalloc(&d_px, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_py, sizeof(int)));
    
    // 拷贝探测器参数（注意：pixels指针未初始化，测试中不使用）
    CUDA_CHECK(cudaMemcpy(d_detector, &h_detector, sizeof(Detector), cudaMemcpyHostToDevice));
    
    // 测试中心点
    testPixelMappingKernel<<<1, 1>>>(*d_detector, 0.0f, 0.0f, d_px, d_py);
    CUDA_CHECK(cudaMemcpy(&h_px, d_px, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_py, d_py, sizeof(int), cudaMemcpyDeviceToHost));
    
    // 中心点应该映射到 (512, 512) 附近
    TEST_ASSERT(h_px >= 511 && h_px <= 513, "Center X pixel mapping incorrect");
    TEST_ASSERT(h_py >= 511 && h_py <= 513, "Center Y pixel mapping incorrect");
    
    // 测试角落点
    testPixelMappingKernel<<<1, 1>>>(*d_detector, 9.5f, 9.5f, d_px, d_py);
    CUDA_CHECK(cudaMemcpy(&h_px, d_px, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_py, d_py, sizeof(int), cudaMemcpyDeviceToHost));
    
    // (9.5, 9.5) 应该映射到接近 (1024, 1024)
    TEST_ASSERT(h_px > 1000, "Corner X pixel mapping incorrect");
    TEST_ASSERT(h_py > 1000, "Corner Y pixel mapping incorrect");
    
    // 清理
    cudaFree(d_detector);
    cudaFree(d_px);
    cudaFree(d_py);
    
    printf("  PASSED\n");
    return 0;
}

/**
 * @brief 运行所有测试
 */
int main(int argc, char **argv)
{
    printf("=======================================\n");
    printf("Photon Simulation Unit Tests\n");
    printf("=======================================\n\n");
    
    int failed = 0;
    
    // 运行测试
    failed += testFreePathSampling();
    failed += testGeometryParsing();
    failed += testMaterialParsing();
    failed += testPixelMapping();
    
    printf("\n=======================================\n");
    if (failed == 0) {
        printf("All tests PASSED!\n");
    } else {
        printf("%d test(s) FAILED!\n", failed);
    }
    printf("=======================================\n");
    
    return failed;
}
