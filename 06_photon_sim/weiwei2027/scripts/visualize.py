#!/usr/bin/env python3
"""
光子传输模拟结果可视化脚本
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np


def read_detector_image(filename, width=None, height=None):
    """
    读取探测器图像二进制文件
    
    参数:
        filename: 图像文件路径
        width: 图像宽度（如果文件不包含元数据）
        height: 图像高度（如果文件不包含元数据）
    
    返回:
        2D numpy数组，形状为 (height, width)
    """
    with open(filename, 'rb') as f:
        # 尝试读取元数据（2个int作为头部）
        if width is None or height is None:
            header = np.fromfile(f, dtype=np.int32, count=2)
            if len(header) >= 2:
                width, height = header[0], header[1]
                print(f"Read header: width={width}, height={height}")
            else:
                raise ValueError("Cannot determine image dimensions")
        else:
            # 如果指定了宽高，跳过头部
            f.seek(8)

        # 读取像素数据
        expected_size = width * height
        pixels = np.fromfile(f, dtype=np.float32)

        if len(pixels) != expected_size:
            print(f"Warning: Expected {expected_size} pixels, got {len(pixels)}")
            # 尝试 reshape 实际读取的数据
            height = len(pixels) // width

    # 重塑为2D数组
    image = pixels.reshape((height, width))
    return image


def read_image_info(filename):
    """
    读取图像信息描述文件
    
    返回:
        dict 包含 width, height, dtype 等信息
    """
    info = {}
    with open(filename, 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            info[key] = value

    # 转换数值类型
    if 'width' in info:
        info['width'] = int(info['width'])
    if 'height' in info:
        info['height'] = int(info['height'])

    return info


def visualize_image(image, title="X-ray Detector Image", cmap='hot', save_path=None):
    """
    可视化探测器图像
    
    参数:
        image: 2D numpy数组
        title: 图像标题
        cmap: 颜色映射
        save_path: 保存路径（可选）
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 线性刻度
    im1 = axes[0].imshow(image, cmap=cmap, origin='lower')
    axes[0].set_title(f"{title} (Linear)")
    axes[0].set_xlabel("Pixel X")
    axes[0].set_ylabel("Pixel Y")
    plt.colorbar(im1, ax=axes[0], label='Photon Count')

    # 对数刻度（更好显示动态范围）
    image_log = np.log1p(image)  # log(1 + x) 避免 log(0)
    im2 = axes[1].imshow(image_log, cmap=cmap, origin='lower')
    axes[1].set_title(f"{title} (Log Scale)")
    axes[1].set_xlabel("Pixel X")
    axes[1].set_ylabel("Pixel Y")
    plt.colorbar(im2, ax=axes[1], label='Log(Photon Count)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Image saved to: {save_path}")

    plt.show()


def plot_profile(image, axis=0, position=None):
    """
    绘制图像的剖面线
    
    参数:
        image: 2D numpy数组
        axis: 0 表示水平剖面，1 表示垂直剖面
        position: 剖面位置（默认为中心）
    """
    if position is None:
        position = image.shape[axis] // 2

    if axis == 0:
        profile = image[position, :]
        x_label = "Pixel X"
        title = f"Horizontal Profile at Y={position}"
    else:
        profile = image[:, position]
        x_label = "Pixel Y"
        title = f"Vertical Profile at X={position}"

    plt.figure(figsize=(10, 4))
    plt.plot(profile)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Photon Count")
    plt.grid(True)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize X-ray detector simulation results')
    parser.add_argument('input', help='Input binary image file')
    parser.add_argument('--info', help='Image info file (if not embedded in binary)')
    parser.add_argument('--width', type=int, help='Image width (if not in file)')
    parser.add_argument('--height', type=int, help='Image height (if not in file)')
    parser.add_argument('--save', help='Save figure to file instead of displaying')
    parser.add_argument('--cmap', default='hot', help='Colormap (default: hot)')
    parser.add_argument('--profile', action='store_true', help='Show profile plots')

    args = parser.parse_args()

    # 读取图像
    try:
        if args.info:
            info = read_image_info(args.info)
            image = read_detector_image(args.input, info['width'], info['height'])
        else:
            image = read_detector_image(args.input, args.width, args.height)
    except Exception as e:
        print(f"Error reading image: {e}")
        sys.exit(1)

    print(f"Image loaded: {image.shape[1]} x {image.shape[0]} pixels")
    print(f"Photon count range: [{image.min():.2f}, {image.max():.2f}]")
    print(f"Total photons detected: {image.sum():.2e}")

    # 可视化
    visualize_image(image, save_path=args.save, cmap=args.cmap)

    # 绘制剖面
    if args.profile:
        plot_profile(image, axis=0)
        plot_profile(image, axis=1)


if __name__ == '__main__':
    main()
