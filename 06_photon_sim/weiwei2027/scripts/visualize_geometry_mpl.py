#!/usr/bin/env python3
"""
医学成像光子传输模拟 - 3D 几何可视化 (Matplotlib版本)
使用 Matplotlib 的 3D 绘图功能，无需 PyVista

安装依赖:
    pip install numpy matplotlib

运行:
    python3 visualize_geometry_mpl.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches


def draw_cuboid(ax, x, y, z, dx, dy, dz, color, alpha=0.5, edge_color='black'):
    """绘制长方体"""
    # 定义8个顶点
    vertices = [
        [x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],  # 底面
        [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]  # 顶面
    ]
    
    # 定义6个面
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
        [vertices[3], vertices[0], vertices[4], vertices[7]],  # 左面
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
    ]
    
    # 创建3D面集合
    face_collection = Poly3DCollection(faces, alpha=alpha, 
                                       facecolor=color, 
                                       edgecolor=edge_color,
                                       linewidth=0.5)
    ax.add_collection3d(face_collection)


def draw_sphere(ax, center, radius, color='red', alpha=0.6, num_points=20):
    """绘制球体"""
    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, np.pi, num_points)
    
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x, y, z, color=color, alpha=alpha, 
                   edgecolor='none', shade=True)
    
    # 绘制线框
    ax.plot_wireframe(x, y, z, color='darkred', alpha=0.3, 
                     rstride=4, cstride=4, linewidth=0.5)


def draw_detector(ax, z_pos, width=20, height=20, color='blue', alpha=0.3):
    """绘制探测器平面"""
    # 绘制矩形平面
    x = np.array([[-width/2, -width/2], [width/2, width/2]])
    y = np.array([[-height/2, height/2], [-height/2, height/2]])
    z = np.array([[z_pos, z_pos], [z_pos, z_pos]])
    
    ax.plot_surface(x, y, z, color=color, alpha=alpha, 
                   edgecolor='blue', linewidth=2)
    
    # 绘制网格线
    for i in range(-5, 6):
        x_line = [i*width/10, i*width/10]
        y_line = [-height/2, height/2]
        z_line = [z_pos, z_pos]
        ax.plot(x_line, y_line, z_line, 'b-', alpha=0.3, linewidth=0.5)
        
        x_line = [-width/2, width/2]
        y_line = [i*height/10, i*height/10]
        z_line = [z_pos, z_pos]
        ax.plot(x_line, y_line, z_line, 'b-', alpha=0.3, linewidth=0.5)


def draw_photon_rays(ax, source_pos, num_rays=10, spread=2.0, length=18):
    """绘制示例光子射线"""
    np.random.seed(42)
    
    for i in range(num_rays):
        # 起始位置
        start_x = np.random.uniform(-spread, spread)
        start_y = np.random.uniform(-spread, spread)
        
        # 绘制射线
        x = [start_x, start_x]
        y = [start_y, start_y]
        z = [source_pos[2], source_pos[2] + length]
        
        color = 'green' if i % 2 == 0 else 'lime'
        ax.plot(x, y, z, color=color, linewidth=1.5, alpha=0.6)


def create_3d_visualization():
    """创建3D可视化"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # ==================== 几何参数 ====================
    # 平板层: (name, z_start, thickness, color, alpha)
    layers = [
        ('skin', 0.0, 0.2, 'sandybrown', 0.5),
        ('skull', 0.2, 0.8, 'ivory', 0.7),
        ('brain', 1.0, 16.0, 'lightpink', 0.3),
    ]
    
    # 球体: (name, center, radius, color)
    spheres = [
        ('hematoma', [0.0, 0.0, 5.0], 0.5, 'darkred'),
    ]
    
    # 源和探测器位置
    source_pos = [0.0, 0.0, -1.0]
    detector_z = 17.5
    
    width = height = 20  # 层的宽度和高度
    
    # ==================== 绘制平板层 ====================
    print("Drawing layers...")
    for name, z_start, thickness, color, alpha in layers:
        draw_cuboid(ax, -width/2, -height/2, z_start, 
                   width, height, thickness, 
                   color=color, alpha=alpha)
        
        # 添加标签
        label_x = width/2 + 1
        label_y = 0
        label_z = z_start + thickness/2
        ax.text(label_x, label_y, label_z, 
               f'{name}\n({thickness}cm)', 
               fontsize=8, color='black')
    
    # ==================== 绘制球体 ====================
    print("Drawing spheres...")
    for name, center, radius, color in spheres:
        draw_sphere(ax, center, radius, color=color, alpha=0.7)
        
        # 添加标签
        ax.text(center[0] + radius + 1, center[1], center[2],
               f'{name}\nr={radius}cm',
               fontsize=8, color='darkred')
    
    # ==================== 绘制探测器 ====================
    print("Drawing detector...")
    draw_detector(ax, detector_z, width=20, height=20)
    ax.text(11, 0, detector_z, 'Detector\n(1024×1024)',
           fontsize=8, color='blue')
    
    # ==================== 绘制光源 ====================
    print("Drawing source...")
    # 绘制光源点
    ax.scatter([source_pos[0]], [source_pos[1]], [source_pos[2]],
              color='gold', s=200, marker='o', edgecolors='orange',
              linewidths=2, label='X-ray Source')
    ax.text(source_pos[0] + 2, source_pos[1], source_pos[2],
           'X-ray Source', fontsize=9, color='orange')
    
    # ==================== 绘制光子射线 ====================
    print("Drawing photon rays...")
    draw_photon_rays(ax, source_pos, num_rays=15)
    
    # ==================== 绘制坐标轴和标注 ====================
    # 添加Z轴箭头表示方向
    ax.quiver(0, -12, 0, 0, 0, 10, color='green', 
             arrow_length_ratio=0.3, linewidth=2)
    ax.text(0, -12, 5, 'Photon\nDirection', fontsize=8, color='green')
    
    # 添加尺寸线
    ax.plot([12, 12], [-10, -10], [0, 17.5], 'gray', linewidth=2)
    ax.text(12, -10, 8.75, '17.5 cm', fontsize=8, color='gray')
    
    # ==================== 设置视角和标签 ====================
    ax.set_xlabel('X (cm)', fontsize=10)
    ax.set_ylabel('Y (cm)', fontsize=10)
    ax.set_zlabel('Z (cm)', fontsize=10)
    
    ax.set_xlim(-12, 15)
    ax.set_ylim(-12, 15)
    ax.set_zlim(-3, 20)
    
    # 设置等比例
    ax.set_box_aspect([1, 1, 1.5])
    
    # 设置视角
    ax.view_init(elev=15, azim=-60)
    
    # 标题
    plt.title('Medical Imaging Photon Transport Simulation\n'
             'Layered Head Model with Sphere Inclusion',
             fontsize=12, fontweight='bold')
    
    # 图例
    legend_elements = [
        mpatches.Patch(facecolor='sandybrown', edgecolor='black', 
                      label='Skin (0.2cm)', alpha=0.5),
        mpatches.Patch(facecolor='ivory', edgecolor='black',
                      label='Skull (0.8cm)', alpha=0.7),
        mpatches.Patch(facecolor='lightpink', edgecolor='black',
                      label='Brain (16cm)', alpha=0.3),
        mpatches.Patch(facecolor='darkred', edgecolor='black',
                      label='Hematoma (r=0.5cm)', alpha=0.7),
        mpatches.Patch(facecolor='lightblue', edgecolor='blue',
                      label='Detector', alpha=0.3),
        mpatches.Patch(facecolor='gold', edgecolor='orange',
                      label='X-ray Source', alpha=1.0),
    ]
    ax.legend(handles=legend_elements, loc='upper left', 
             bbox_to_anchor=(1.05, 1), fontsize=8)
    
    plt.tight_layout()
    
    # 保存
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'geometry_3d_mpl.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")
    
    if '--no-display' not in sys.argv:
        plt.show()


def create_side_view():
    """创建侧视图（YZ平面）"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 层定义: (z_start, thickness, color, label)
    layers = [
        (0.0, 0.2, 'sandybrown', 'Skin'),
        (0.2, 0.8, 'lightgray', 'Skull'),
        (1.0, 16.0, 'lightpink', 'Brain'),
    ]
    
    # 绘制层
    for z_start, thickness, color, label in layers:
        rect = plt.Rectangle((-10, z_start), 20, thickness,
                            facecolor=color, edgecolor='black',
                            linewidth=1.5, alpha=0.6)
        ax.add_patch(rect)
        ax.text(11, z_start + thickness/2, label, 
               va='center', fontsize=9)
    
    # 绘制球体（侧视图是圆）
    circle = plt.Circle((0, 5), 0.5, color='darkred', 
                       alpha=0.7, label='Hematoma')
    ax.add_patch(circle)
    ax.text(2, 5, 'Hematoma\n(r=0.5cm)', va='center', fontsize=8)
    
    # 探测器
    ax.axhline(y=17.5, color='blue', linewidth=3, 
              label='Detector', alpha=0.7)
    ax.text(11, 17.5, 'Detector', va='center', fontsize=9, color='blue')
    
    # 光源
    ax.scatter([0], [-1], s=200, c='gold', marker='o',
              edgecolors='orange', linewidths=2, zorder=5)
    ax.text(2, -1, 'Source', va='center', fontsize=9, color='orange')
    
    # 光子射线示例
    for i in range(-3, 4):
        ax.arrow(i*0.5, -0.8, 0, 18, head_width=0.3, 
                head_length=0.5, fc='green', ec='green',
                alpha=0.4, linewidth=1)
    
    # 设置坐标轴
    ax.set_xlim(-12, 18)
    ax.set_ylim(-3, 20)
    ax.set_aspect('equal')
    ax.set_xlabel('Y (cm)', fontsize=10)
    ax.set_ylabel('Z (cm)', fontsize=10)
    ax.set_title('Side View (YZ Plane)\nPhoton Transport Geometry',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 添加Z轴方向标注
    ax.annotate('Photon Direction →', xy=(0, 9), xytext=(-8, 9),
               fontsize=10, color='green',
               arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    plt.tight_layout()
    
    # 保存
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'geometry_side_view.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
               facecolor='white')
    print(f"Saved: {output_path}")
    
    if '--no-display' not in sys.argv:
        plt.show()


if __name__ == '__main__':
    import sys
    import os
    
    # 设置无GUI后端（如果需要）
    if os.environ.get('DISPLAY') is None or '--no-display' in sys.argv:
        import matplotlib
        matplotlib.use('Agg')
        print("Using non-interactive backend (Agg)")
    
    if '--side' in sys.argv:
        create_side_view()
    else:
        create_3d_visualization()
