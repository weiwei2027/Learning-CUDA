/**
 * 解析模块单元测试
 * 验证 geometry、material、io 解析的正确性
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>

// 包含被测试的模块
#include "../include/types.h"
#include "../include/utils.h"

// 测试宏
#define TEST(name) printf("\n[TEST] %s\n", name);
#define PASS() printf("  ✓ PASSED\n")
#define FAIL(msg) do { printf("  ✗ FAILED: %s\n", msg); return 1; } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) FAIL(msg); } while(0)
#define ASSERT_FLOAT_EQ(a, b, eps, msg) do { \
    if (fabs((a) - (b)) > (eps)) { \
        printf("  ✗ FAILED: %s (%.6f vs %.6f, diff=%.6f)\n", msg, (float)(a), (float)(b), fabs((a)-(b))); \
        return 1; \
    } \
} while(0)

/**
 * 测试 1: 几何文件解析
 */
int testGeometryParsing()
{
    TEST("Geometry File Parsing");
    
    // 创建临时测试文件
    const char *test_file = "/tmp/test_geometry.txt";
    FILE *fp = fopen(test_file, "w");
    ASSERT(fp != nullptr, "Cannot create test file");
    
    fprintf(fp, "# Test geometry file\n");
    fprintf(fp, "layer skin 0.2 0.0 0.0 0.0 tissue_skin\n");
    fprintf(fp, "layer skull 0.8 0.0 0.0 0.2 tissue_bone\n");
    fprintf(fp, "\n");  // 空行
    fprintf(fp, "layer brain 16.0 0.0 0.0 1.0 tissue_brain\n");
    fprintf(fp, "# sphere hematoma 0.5 5.0 5.0 5.0 tissue_blood\n");  // 注释掉的球体
    fclose(fp);
    
    // 解析
    Region *regions = nullptr;
    int num_regions = 0;
    int ret = parseGeometryFile(test_file, &regions, &num_regions);
    ASSERT(ret == 0, "parseGeometryFile failed");
    ASSERT(num_regions == 3, "Should have 3 regions");
    
    // 验证第一层
    ASSERT(regions[0].type == LAYER, "Region 0 should be LAYER");
    ASSERT(strcmp(regions[0].name, "skin") == 0, "Region 0 name should be 'skin'");
    ASSERT_FLOAT_EQ(regions[0].thickness, 0.2f, 1e-6f, "Region 0 thickness should be 0.2");
    ASSERT_FLOAT_EQ(regions[0].z, 0.0f, 1e-6f, "Region 0 z should be 0.0");
    ASSERT(strcmp(regions[0].material_name, "tissue_skin") == 0, "Region 0 material should be 'tissue_skin'");
    
    // 验证第二层
    ASSERT(regions[1].type == LAYER, "Region 1 should be LAYER");
    ASSERT(strcmp(regions[1].name, "skull") == 0, "Region 1 name should be 'skull'");
    ASSERT_FLOAT_EQ(regions[1].thickness, 0.8f, 1e-6f, "Region 1 thickness should be 0.8");
    ASSERT_FLOAT_EQ(regions[1].z, 0.2f, 1e-6f, "Region 1 z should be 0.2");
    
    // 验证第三层
    ASSERT(regions[2].type == LAYER, "Region 2 should be LAYER");
    ASSERT(strcmp(regions[2].name, "brain") == 0, "Region 2 name should be 'brain'");
    ASSERT_FLOAT_EQ(regions[2].thickness, 16.0f, 1e-6f, "Region 2 thickness should be 16.0");
    ASSERT_FLOAT_EQ(regions[2].z, 1.0f, 1e-6f, "Region 2 z should be 1.0");
    
    // 验证连续性（层是否连续）
    float end_of_skin = regions[0].z + regions[0].thickness;
    ASSERT_FLOAT_EQ(end_of_skin, regions[1].z, 1e-6f, "Skin should end where skull begins");
    
    float end_of_skull = regions[1].z + regions[1].thickness;
    ASSERT_FLOAT_EQ(end_of_skull, regions[2].z, 1e-6f, "Skull should end where brain begins");
    
    printf("  Geometry validation:\n");
    printf("    - Region count: %d ✓\n", num_regions);
    printf("    - Layer continuity: ✓\n");
    printf("    - Total thickness: %.1f cm ✓\n", regions[2].z + regions[2].thickness);
    
    // 清理
    free(regions);
    
    PASS();
    return 0;
}

/**
 * 测试 2: 材料文件解析
 */
int testMaterialParsing()
{
    TEST("Material File Parsing");
    
    const char *test_file = "/tmp/test_materials.csv";
    FILE *fp = fopen(test_file, "w");
    ASSERT(fp != nullptr, "Cannot create test file");
    
    fprintf(fp, "material,energy_keV,mu\n");
    fprintf(fp, "tissue_skin,50,0.2\n");
    fprintf(fp, "tissue_skin,100,0.15\n");
    fprintf(fp, "tissue_bone,50,0.5\n");
    fprintf(fp, "tissue_bone,100,0.3\n");
    fclose(fp);
    
    Material *materials = nullptr;
    int num_materials = 0;
    int ret = parseMaterialFile(test_file, &materials, &num_materials);
    ASSERT(ret == 0, "parseMaterialFile failed");
    ASSERT(num_materials == 4, "Should have 4 material entries");
    
    // 验证第一行
    ASSERT(strcmp(materials[0].name, "tissue_skin") == 0, "Material 0 name should be 'tissue_skin'");
    ASSERT_FLOAT_EQ(materials[0].energy, 50.0f, 1e-6f, "Material 0 energy should be 50");
    ASSERT_FLOAT_EQ(materials[0].mu, 0.2f, 1e-6f, "Material 0 mu should be 0.2");
    
    // 验证第三行
    ASSERT(strcmp(materials[2].name, "tissue_bone") == 0, "Material 2 name should be 'tissue_bone'");
    ASSERT_FLOAT_EQ(materials[2].energy, 50.0f, 1e-6f, "Material 2 energy should be 50");
    ASSERT_FLOAT_EQ(materials[2].mu, 0.5f, 1e-6f, "Material 2 mu should be 0.5");
    
    printf("  Material validation:\n");
    printf("    - Entry count: %d ✓\n", num_materials);
    printf("    - Skin at 50keV: mu=%.2f ✓\n", materials[0].mu);
    printf("    - Bone at 50keV: mu=%.2f ✓\n", materials[2].mu);
    
    free(materials);
    
    PASS();
    return 0;
}

/**
 * 测试 3: 材料插值
 */
int testMaterialInterpolation()
{
    TEST("Material Energy Interpolation");
    
    // 创建测试材料数据
    Material materials[4];
    strcpy(materials[0].name, "tissue_test");
    materials[0].energy = 50.0f;
    materials[0].mu = 0.2f;
    
    strcpy(materials[1].name, "tissue_test");
    materials[1].energy = 100.0f;
    materials[1].mu = 0.15f;
    
    strcpy(materials[2].name, "tissue_other");
    materials[2].energy = 50.0f;
    materials[2].mu = 0.5f;
    
    strcpy(materials[3].name, "tissue_other");
    materials[3].energy = 100.0f;
    materials[3].mu = 0.3f;
    
    // 测试精确匹配
    float mu_50 = interpolateMu(materials, 4, "tissue_test", 50.0f);
    ASSERT_FLOAT_EQ(mu_50, 0.2f, 1e-6f, "Exact match at 50keV failed");
    
    float mu_100 = interpolateMu(materials, 4, "tissue_test", 100.0f);
    ASSERT_FLOAT_EQ(mu_100, 0.15f, 1e-6f, "Exact match at 100keV failed");
    
    // 测试插值（75keV 应该在 0.2 和 0.15 中间）
    float mu_75 = interpolateMu(materials, 4, "tissue_test", 75.0f);
    float expected_75 = (0.2f + 0.15f) / 2.0f;  // 线性插值
    ASSERT_FLOAT_EQ(mu_75, expected_75, 1e-6f, "Interpolation at 75keV failed");
    
    printf("  Interpolation validation:\n");
    printf("    - Exact match (50keV): mu=%.4f ✓\n", mu_50);
    printf("    - Exact match (100keV): mu=%.4f ✓\n", mu_100);
    printf("    - Interpolation (75keV): mu=%.4f (expected %.4f) ✓\n", mu_75, expected_75);
    
    PASS();
    return 0;
}

/**
 * 测试 4: 源参数解析
 */
int testSourceParsing()
{
    TEST("Source Configuration Parsing");
    
    const char *test_file = "/tmp/test_source.txt";
    FILE *fp = fopen(test_file, "w");
    ASSERT(fp != nullptr, "Cannot create test file");
    
    fprintf(fp, "# Source configuration\n");
    fprintf(fp, "source_type = point\n");
    fprintf(fp, "position = 1.0 2.0 -1.5\n");
    fprintf(fp, "direction = 0.0 0.0 1.0\n");
    fprintf(fp, "energy = 75.0\n");
    fprintf(fp, "num_photons = 1000000\n");
    fprintf(fp, "\n");
    fprintf(fp, "detector_position = 0.0 0.0 17.5\n");
    fprintf(fp, "detector_size = 20.0 20.0\n");
    fprintf(fp, "detector_pixels = 512 512\n");
    fclose(fp);
    
    Source source;
    Detector detector;
    int ret = parseSourceFile(test_file, &source, &detector);
    ASSERT(ret == 0, "parseSourceFile failed");
    
    // 验证源参数
    ASSERT(source.type == 0, "Source type should be point (0)");
    ASSERT_FLOAT_EQ(source.x, 1.0f, 1e-6f, "Source x should be 1.0");
    ASSERT_FLOAT_EQ(source.y, 2.0f, 1e-6f, "Source y should be 2.0");
    ASSERT_FLOAT_EQ(source.z, -1.5f, 1e-6f, "Source z should be -1.5");
    ASSERT_FLOAT_EQ(source.energy, 75.0f, 1e-6f, "Source energy should be 75.0");
    ASSERT(source.num_photons == 1000000, "Number of photons should be 1000000");
    
    // 验证探测器参数
    ASSERT_FLOAT_EQ(detector.z, 17.5f, 1e-6f, "Detector z should be 17.5");
    ASSERT_FLOAT_EQ(detector.width, 20.0f, 1e-6f, "Detector width should be 20.0");
    ASSERT_FLOAT_EQ(detector.height, 20.0f, 1e-6f, "Detector height should be 20.0");
    ASSERT(detector.nx == 512, "Detector nx should be 512");
    ASSERT(detector.ny == 512, "Detector ny should be 512");
    
    printf("  Source validation:\n");
    printf("    - Type: %s ✓\n", source.type == 0 ? "Point" : "Parallel");
    printf("    - Position: (%.1f, %.1f, %.1f) ✓\n", source.x, source.y, source.z);
    printf("    - Energy: %.1f keV ✓\n", source.energy);
    printf("    - Photons: %d ✓\n", source.num_photons);
    printf("  Detector validation:\n");
    printf("    - Position: z=%.1f cm ✓\n", detector.z);
    printf("    - Size: %.1f x %.1f cm ✓\n", detector.width, detector.height);
    printf("    - Resolution: %d x %d pixels ✓\n", detector.nx, detector.ny);
    
    PASS();
    return 0;
}

/**
 * 测试 5: 边界条件测试
 */
int testEdgeCases()
{
    TEST("Edge Cases");
    
    // 测试 1: 空几何文件
    {
        const char *empty_file = "/tmp/test_empty_geo.txt";
        FILE *fp = fopen(empty_file, "w");
        fprintf(fp, "# Only comments\n");
        fprintf(fp, "\n");
        fprintf(fp, "# No actual regions\n");
        fclose(fp);
        
        Region *regions = nullptr;
        int num_regions = 0;
        int ret = parseGeometryFile(empty_file, &regions, &num_regions);
        ASSERT(ret == 0, "Should handle empty file without error");
        ASSERT(num_regions == 0, "Should have 0 regions");
        
        printf("  - Empty geometry file: handled ✓\n");
    }
    
    // 测试 2: 只有一个区域
    {
        const char *single_file = "/tmp/test_single_geo.txt";
        FILE *fp = fopen(single_file, "w");
        fprintf(fp, "layer single 5.0 0.0 0.0 0.0 tissue_test\n");
        fclose(fp);
        
        Region *regions = nullptr;
        int num_regions = 0;
        int ret = parseGeometryFile(single_file, &regions, &num_regions);
        ASSERT(ret == 0, "Should parse single region");
        ASSERT(num_regions == 1, "Should have 1 region");
        ASSERT(strcmp(regions[0].name, "single") == 0, "Region name should be 'single'");
        
        free(regions);
        printf("  - Single region file: parsed ✓\n");
    }
    
    PASS();
    return 0;
}

/**
 * 主函数
 */
int main(int argc, char **argv)
{
    printf("========================================\n");
    printf("Parser Module Unit Tests\n");
    printf("========================================\n");
    
    int failed = 0;
    
    failed += testGeometryParsing();
    failed += testMaterialParsing();
    failed += testMaterialInterpolation();
    failed += testSourceParsing();
    failed += testEdgeCases();
    
    printf("\n========================================\n");
    if (failed == 0) {
        printf("All tests PASSED! ✓\n");
    } else {
        printf("%d test(s) FAILED! ✗\n", failed);
    }
    printf("========================================\n");
    
    return failed;
}
