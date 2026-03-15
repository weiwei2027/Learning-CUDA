import numpy as np
import matplotlib.pyplot as plt
import sys

def load_image(bin_file):
    """加载二进制图像文件"""
    with open(bin_file, 'rb') as f:
        nx = np.frombuffer(f.read(4), dtype=np.int32)[0]
        ny = np.frombuffer(f.read(4), dtype=np.int32)[0]
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape((ny, nx))

# 加载测试结果
try:
    img1 = load_image('output/test1_point_no_sphere.bin')
    img3 = load_image('output/test3_point_with_sphere.bin')
    img4 = load_image('output/test4_parallel_with_sphere.bin')
except Exception as e:
    print(f"Error loading images: {e}")
    sys.exit(1)

# 计算穿透率
total_photons = 1000000
rate1 = img1.sum() / total_photons * 100
rate3 = img3.sum() / total_photons * 100
rate4 = img4.sum() / total_photons * 100

print("=" * 60)
print("CPU 源模式测试结果对比")
print("=" * 60)
print(f"\n1. Point mode (no sphere):     {rate1:.4f}%  ({img1.sum():.0f} photons)")
print(f"2. Point mode (with sphere):   {rate3:.4f}%  ({img3.sum():.0f} photons)")
print(f"3. Parallel mode (with sphere): {rate4:.4f}%  ({img4.sum():.0f} photons)")
print(f"\n球体效果:")
print(f"   Point mode:   {rate3-rate1:+.4f}% ({img3.sum()-img1.sum():+.0f} photons)")
print(f"   Parallel mode: {rate4-rate1:+.4f}% ({img4.sum()-img1.sum():+.0f} photons)")

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 第一行：原始图像
axes[0, 0].imshow(img1, cmap='hot', extent=[-10, 10, -10, 10])
axes[0, 0].set_title(f'1. Point (no sphere)\n{rate1:.4f}%')
axes[0, 0].set_xlabel('X (cm)')
axes[0, 0].set_ylabel('Y (cm)')

axes[0, 1].imshow(img3, cmap='hot', extent=[-10, 10, -10, 10])
axes[0, 1].set_title(f'2. Point (with sphere)\n{rate3:.4f}%')
axes[0, 1].set_xlabel('X (cm)')

axes[0, 2].imshow(img4, cmap='hot', extent=[-10, 10, -10, 10])
axes[0, 2].set_title(f'3. Parallel (with sphere)\n{rate4:.4f}%')
axes[0, 2].set_xlabel('X (cm)')

# 第二行：差值图
diff3 = img3 - img1
diff4 = img4 - img1

im1 = axes[1, 0].imshow(diff3, cmap='RdBu_r', extent=[-10, 10, -10, 10])
axes[1, 0].set_title(f'Point: Diff (2-1)\nTotal diff: {diff3.sum():.0f}')
axes[1, 0].set_xlabel('X (cm)')
axes[1, 0].set_ylabel('Y (cm)')
plt.colorbar(im1, ax=axes[1, 0])

im2 = axes[1, 1].imshow(diff4, cmap='RdBu_r', extent=[-10, 10, -10, 10])
axes[1, 1].set_title(f'Parallel: Diff (3-1)\nTotal diff: {diff4.sum():.0f}')
axes[1, 1].set_xlabel('X (cm)')
plt.colorbar(im2, ax=axes[1, 1])

# 截面图（通过中心）
center = img1.shape[0] // 2
x_axis = np.linspace(-10, 10, img1.shape[1])
axes[1, 2].plot(x_axis, img1[center, :], 'b-', label='Point (no sphere)', linewidth=2)
axes[1, 2].plot(x_axis, img3[center, :], 'r-', label='Point (with sphere)', linewidth=2)
axes[1, 2].plot(x_axis, img4[center, :], 'g-', label='Parallel (with sphere)', linewidth=2)
axes[1, 2].set_xlabel('X (cm)')
axes[1, 2].set_ylabel('Photon count')
axes[1, 2].set_title('Cross-section at Y=0')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/source_mode_comparison.png', dpi=150)
print(f"\n可视化结果已保存: output/source_mode_comparison.png")

# 检查球体阴影区域（中心区域）
# 球体在 (0,0,5)，投影到探测器上应该在中心附近
# 探测器 1024x1024，范围 [-10, 10] cm，每个像素约 0.02 cm
# 球体半径 0.5 cm，投影大约 0.5 * (17.5-(-1))/(5-(-1)) ≈ 1.5 cm 范围
# 对应约 75 像素

sphere_proj_radius = 0.5 * (17.5 - (-1.0)) / (5.0 - (-1.0))  # 几何放大
print(f"\n球体投影半径估计: {sphere_proj_radius:.2f} cm")

# 中心区域统计
center_region = 50  # 像素范围
x0, y0 = img1.shape[1]//2, img1.shape[0]//2
region1 = img1[y0-center_region:y0+center_region, x0-center_region:x0+center_region]
region3 = img3[y0-center_region:y0+center_region, x0-center_region:x0+center_region]
region4 = img4[y0-center_region:y0+center_region, x0-center_region:x0+center_region]

print(f"\n中心区域 ({center_region*2}x{center_region*2} pixels) 统计:")
print(f"  Point (no sphere):   {region1.sum():.0f} photons")
print(f"  Point (with sphere): {region3.sum():.0f} photons ({region3.sum()-region1.sum():+.0f})")
print(f"  Parallel (sphere):   {region4.sum():.0f} photons ({region4.sum()-region1.sum():+.0f})")
