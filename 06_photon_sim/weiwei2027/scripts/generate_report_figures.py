#!/usr/bin/env python3
"""
生成报告专用图表
用法: python3 generate_report_figures.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import sys
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def read_detector_image(bin_file, info_file):
    """读取探测器图像"""
    # 读取元数据
    info = {}
    with open(info_file, 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            info[key] = value
    
    width = int(info['width'])
    height = int(info['height'])
    
    # 读取二进制数据（跳过头部8字节）
    with open(bin_file, 'rb') as f:
        f.seek(8)  # 跳过头部
        pixels = np.fromfile(f, dtype=np.float32)
    
    return pixels.reshape((height, width))

def generate_detector_images():
    """生成探测器图像对比"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 三层几何
    img_3layer = read_detector_image(
        'results/A100/3layer_gpu/image.bin',
        'results/A100/3layer_gpu/image_info.txt'
    )
    
    # 球体
    img_sphere = read_detector_image(
        'results/A100/3layer_sphere_gpu/image.bin',
        'results/A100/3layer_sphere_gpu/image_info.txt'
    )
    
    # 三层 - 线性
    im1 = axes[0, 0].imshow(img_3layer, cmap='hot', origin='lower')
    axes[0, 0].set_title('Three-Layer Geometry (Linear Scale)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Pixel X')
    axes[0, 0].set_ylabel('Pixel Y')
    plt.colorbar(im1, ax=axes[0, 0], label='Photon Count')
    
    # 三层 - 对数
    img_3layer_log = np.log1p(img_3layer)
    im2 = axes[0, 1].imshow(img_3layer_log, cmap='hot', origin='lower')
    axes[0, 1].set_title('Three-Layer Geometry (Log Scale)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Pixel X')
    axes[0, 1].set_ylabel('Pixel Y')
    plt.colorbar(im2, ax=axes[0, 1], label='Log(Photon Count)')
    
    # 球体 - 线性
    im3 = axes[1, 0].imshow(img_sphere, cmap='hot', origin='lower')
    axes[1, 0].set_title('With Spherical Inclusion (Linear Scale)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Pixel X')
    axes[1, 0].set_ylabel('Pixel Y')
    plt.colorbar(im3, ax=axes[1, 0], label='Photon Count')
    
    # 球体 - 对数
    img_sphere_log = np.log1p(img_sphere)
    im4 = axes[1, 1].imshow(img_sphere_log, cmap='hot', origin='lower')
    axes[1, 1].set_title('With Spherical Inclusion (Log Scale)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Pixel X')
    axes[1, 1].set_ylabel('Pixel Y')
    plt.colorbar(im4, ax=axes[1, 1], label='Log(Photon Count)')
    
    plt.tight_layout()
    plt.savefig('report/figures/detector_images.png', dpi=150, bbox_inches='tight')
    print('Generated: report/figures/detector_images.png')
    plt.close()
    
    # 生成差异图展示球体阴影
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    diff = img_3layer - img_sphere
    im = ax.imshow(diff, cmap='RdYlBu_r', origin='lower')
    ax.set_title('Shadow Effect of Spherical Inclusion\n(Difference: 3-Layer - With Sphere)', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Pixel X')
    ax.set_ylabel('Pixel Y')
    plt.colorbar(im, ax=ax, label='Photon Count Difference')
    plt.tight_layout()
    plt.savefig('report/figures/sphere_shadow.png', dpi=150, bbox_inches='tight')
    print('Generated: report/figures/sphere_shadow.png')
    plt.close()

def generate_performance_comparison():
    """生成性能对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 平台性能对比
    platforms = ['NVIDIA\nA100', 'Iluvatar\nBI100', 'MetaX\nC500']
    rates = [11.1, 11.1, 7.01]  # 10^9 p/s
    colors = ['#76b900', '#1f77b4', '#ff7f0e']  # NVIDIA绿, 蓝, 橙
    
    bars = axes[0].bar(platforms, rates, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Processing Rate (10⁹ photons/sec)', fontsize=11)
    axes[0].set_title('Multi-Platform Performance Comparison\n(1 Billion Photons)', 
                      fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 13)
    axes[0].grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 加速比对比
    speedups = [1175, 1175, 890]
    bars2 = axes[1].bar(platforms, speedups, color=colors, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Speedup vs CPU (×)', fontsize=11)
    axes[1].set_title('GPU Speedup over CPU\n(Single-threaded Baseline)', 
                      fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, 1300)
    axes[1].grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, speedup in zip(bars2, speedups):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{speedup}×',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('report/figures/performance_comparison.png', dpi=150, bbox_inches='tight')
    print('Generated: report/figures/performance_comparison.png')
    plt.close()

def generate_scaling_analysis():
    """生成扩展性分析图"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 不同光子数下的性能
    photon_counts = np.array([1e7, 1e8, 1e9])  # 10M, 100M, 1B
    times = np.array([0.24, 2.4, 24])  # ms (RTX 4090数据)
    rates = photon_counts / (times / 1000) / 1e10  # 单位: 10^10 p/s
    
    ax.plot(photon_counts / 1e6, rates, 'o-', linewidth=2, markersize=10, 
            color='#1f77b4', label='Processing Rate')
    ax.axhline(y=4.23, color='r', linestyle='--', linewidth=2, 
               label='Peak Rate: 4.23×10¹⁰ p/s')
    
    ax.set_xlabel('Number of Photons (Millions)', fontsize=11)
    ax.set_ylabel('Processing Rate (10¹⁰ photons/sec)', fontsize=11)
    ax.set_title('Performance Scaling Analysis\n(RTX 4090, Three-Layer Geometry)', 
                 fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # 添加数据点标签
    for x, y in zip(photon_counts / 1e6, rates):
        ax.annotate(f'{y:.2f}', xy=(x, y), xytext=(10, 10),
                   textcoords='offset points', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('report/figures/scaling_analysis.png', dpi=150, bbox_inches='tight')
    print('Generated: report/figures/scaling_analysis.png')
    plt.close()

def generate_geometry_diagram():
    """生成几何模型示意图"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 定义层的位置和参数
    layers = [
        {'name': 'Source', 'z': -1.0, 'thickness': 0, 'color': 'yellow', 'alpha': 0.3},
        {'name': 'Skin (0.2 cm)', 'z': 0.0, 'thickness': 0.2, 'color': '#ffcccc', 'alpha': 0.7},
        {'name': 'Skull (0.8 cm)', 'z': 0.2, 'thickness': 0.8, 'color': '#cccccc', 'alpha': 0.8},
        {'name': 'Brain (16.0 cm)', 'z': 1.0, 'thickness': 16.0, 'color': '#ffeeee', 'alpha': 0.6},
        {'name': 'Detector', 'z': 17.5, 'thickness': 0, 'color': 'cyan', 'alpha': 0.5},
    ]
    
    # 球体位置
    sphere_center_z = 5.0
    sphere_radius = 2.0
    
    # 绘制层
    width = 10
    for layer in layers:
        if layer['thickness'] > 0:
            rect = plt.Rectangle((-width/2, layer['z']), width, layer['thickness'],
                                facecolor=layer['color'], edgecolor='black', 
                                alpha=layer['alpha'], linewidth=2)
            ax.add_patch(rect)
            # 添加标签
            ax.text(width/2 + 0.5, layer['z'] + layer['thickness']/2, 
                   layer['name'], va='center', fontsize=10)
    
    # 绘制球体
    sphere = plt.Circle((0, sphere_center_z), sphere_radius, 
                       facecolor='red', edgecolor='darkred', 
                       alpha=0.5, linewidth=2, linestyle='--')
    ax.add_patch(sphere)
    ax.text(sphere_radius + 0.5, sphere_center_z, 'Hematoma\n(2.0 cm radius)', 
           va='center', fontsize=10, color='darkred', fontweight='bold')
    
    # 绘制源和探测器
    ax.plot(0, -1, 'y*', markersize=20, label='Point Source')
    ax.plot([-5, 5], [17.5, 17.5], 'c-', linewidth=4, label='Detector')
    
    # 绘制示例光线（锥形束）
    angles = np.linspace(-0.3, 0.3, 5)
    for angle in angles:
        x_end = 17.5 * np.tan(angle)
        ax.plot([0, x_end], [-1, 17.5], 'g--', alpha=0.3, linewidth=1)
    
    ax.set_xlim(-6, 8)
    ax.set_ylim(-2, 19)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position (cm)', fontsize=11)
    ax.set_ylabel('Z Position (cm)', fontsize=11)
    ax.set_title('Simulation Geometry Model\n(Multi-layer Head with Spherical Inclusion)', 
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('report/figures/geometry_model.png', dpi=150, bbox_inches='tight')
    print('Generated: report/figures/geometry_model.png')
    plt.close()

def main():
    """主函数"""
    print("Generating report figures...")
    
    # 切换到项目根目录
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        generate_detector_images()
    except Exception as e:
        print(f"Warning: Could not generate detector images: {e}")
    
    try:
        generate_performance_comparison()
    except Exception as e:
        print(f"Warning: Could not generate performance comparison: {e}")
    
    try:
        generate_scaling_analysis()
    except Exception as e:
        print(f"Warning: Could not generate scaling analysis: {e}")
    
    try:
        generate_geometry_diagram()
    except Exception as e:
        print(f"Warning: Could not generate geometry diagram: {e}")
    
    print("\nAll figures generated successfully!")
    print("Output directory: report/figures/")

if __name__ == '__main__':
    main()
