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
    img1 = load_image('output/test1_point_fixed.bin')
    img2 = load_image('output/test2_point_sphere_fixed.bin')
    img3 = load_image('output/test3_parallel_sphere_fixed.bin')
    img4 = load_image('output/test4_parallel_no_sphere.bin')
except Exception as e:
    print(f"Error loading images: {e}")
    sys.exit(1)

# 计算穿透率
total_photons = 1000000
rate1 = img1.sum() / total_photons * 100
rate2 = img2.sum() / total_photons * 100
rate3 = img3.sum() / total_photons * 100
rate4 = img4.sum() / total_photons * 100

print("=" * 70)
print("CPU 源模式测试结果对比（修复后）")
print("=" * 70)
print(f"\n1. Point mode (no sphere):      {rate1:.4f}%  ({img1.sum():.0f} photons)")
print(f"2. Point mode (with sphere):    {rate2:.4f}%  ({img2.sum():.0f} photons)")
print(f"3. Parallel mode (no sphere):   {rate4:.4f}%  ({img4.sum():.0f} photons)")
print(f"4. Parallel mode (with sphere): {rate3:.4f}%  ({img3.sum():.0f} photons)")

print(f"\n球体效果:")
print(f"   Point mode:   {rate2-rate1:+.4f}% ({img2.sum()-img1.sum():+.0f} photons)")
print(f"   Parallel mode: {rate3-rate4:+.4f}% ({img3.sum()-img4.sum():+.0f} photons)")

# 可视化
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

images = [img1, img2, img4, img3]
titles = [
    f'1. Point (no sphere)\n{rate1:.4f}%',
    f'2. Point (with sphere)\n{rate2:.4f}%',
    f'3. Parallel (no sphere)\n{rate4:.4f}%',
    f'4. Parallel (with sphere)\n{rate3:.4f}%'
]

for i, (img, title) in enumerate(zip(images, titles)):
    axes[0, i].imshow(img, cmap='hot', extent=[-10, 10, -10, 10])
    axes[0, i].set_title(title)
    axes[0, i].set_xlabel('X (cm)')
    if i == 0:
        axes[0, i].set_ylabel('Y (cm)')

# 差值图
diff_point = img2 - img1
diff_parallel = img3 - img4

im1 = axes[1, 0].imshow(diff_point, cmap='RdBu_r', extent=[-10, 10, -10, 10], vmin=-2, vmax=2)
axes[1, 0].set_title(f'Point: Diff (2-1)\nTotal: {diff_point.sum():.0f}')
axes[1, 0].set_xlabel('X (cm)')
axes[1, 0].set_ylabel('Y (cm)')
plt.colorbar(im1, ax=axes[1, 0])

im2 = axes[1, 1].imshow(diff_parallel, cmap='RdBu_r', extent=[-10, 10, -10, 10], vmin=-2, vmax=2)
axes[1, 1].set_title(f'Parallel: Diff (4-3)\nTotal: {diff_parallel.sum():.0f}')
axes[1, 1].set_xlabel('X (cm)')
plt.colorbar(im2, ax=axes[1, 1])

# 中心区域放大
center = img1.shape[0] // 2
zoom = 100  # 像素
extent_zoom = [-10 * zoom/center, 10 * zoom/center, -10 * zoom/center, 10 * zoom/center]

axes[1, 2].imshow(img2[center-zoom:center+zoom, center-zoom:center+zoom], cmap='hot', extent=extent_zoom)
axes[1, 2].set_title('Point+Sphere (zoom center)')
axes[1, 2].set_xlabel('X (cm)')

axes[1, 3].imshow(img3[center-zoom:center+zoom, center-zoom:center+zoom], cmap='hot', extent=extent_zoom)
axes[1, 3].set_title('Parallel+Sphere (zoom center)')
axes[1, 3].set_xlabel('X (cm)')

plt.tight_layout()
plt.savefig('output/source_mode_fixed_comparison.png', dpi=150)
print(f"\n可视化结果已保存: output/source_mode_fixed_comparison.png")

# 中心区域统计
center_region = 50  # 像素范围
x0, y0 = img1.shape[1]//2, img1.shape[0]//2

print(f"\n中心区域 ({center_region*2}x{center_region*2} pixels) 统计:")
for name, img in [('Point no sphere', img1), ('Point sphere', img2), 
                   ('Parallel no sphere', img4), ('Parallel sphere', img3)]:
    region = img[y0-center_region:y0+center_region, x0-center_region:x0+center_region]
    print(f"  {name:20s}: {region.sum():.0f} photons")
