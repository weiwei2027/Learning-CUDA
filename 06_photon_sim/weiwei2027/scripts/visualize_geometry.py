#!/usr/bin/env python3
"""
医学成像光子传输模拟 - 3D 几何可视化
使用 PyVista 展示头部模型、球体异物、光源和探测器

安装依赖:
    pip install pyvista numpy matplotlib

运行:
    python3 visualize_geometry.py
"""

import numpy as np
import pyvista as pv

# 设置中文字体（如果可用）
pv.global_theme.font.family = 'arial'


def create_layer_mesh(z_start, thickness, width=20, height=20, color='tan'):
    """创建一个平板层（立方体）"""
    # 创建长方体
    center = [0, 0, z_start + thickness / 2]
    mesh = pv.Cube(center=center, x_length=width, y_length=height, z_length=thickness)
    return mesh


def create_sphere_mesh(center, radius, color='red', opacity=0.7):
    """创建球体"""
    sphere = pv.Sphere(radius=radius, center=center)
    return sphere


def create_detector_mesh(z_pos, width=20, height=20, color='blue'):
    """创建探测器平面"""
    # 创建平面
    plane = pv.Plane(center=[0, 0, z_pos], direction=[0, 0, 1],
                     i_size=width, j_size=height)
    return plane


def create_source_marker(position, color='yellow'):
    """创建光源标记"""
    sphere = pv.Sphere(radius=0.3, center=position)
    return sphere


def create_photon_rays(source_pos, num_rays=20, spread=2.0, length=20):
    """创建示例光子射线"""
    lines = []
    
    # 随机生成一些光子路径
    np.random.seed(42)
    for i in range(num_rays):
        # 起始位置（带随机偏移）
        start_x = np.random.uniform(-spread, spread)
        start_y = np.random.uniform(-spread, spread)
        start = [start_x, start_y, source_pos[2]]
        
        # 结束位置（沿Z轴）
        end = [start_x, start_y, source_pos[2] + length]
        
        # 创建线段
        line = pv.Line(start, end)
        lines.append(line)
    
    return lines


def main():
    """主可视化函数"""
    
    # 创建绘图器
    plotter = pv.Plotter(off_screen=False)
    plotter.set_background('white')
    plotter.add_axes()
    plotter.show_grid()
    
    # ==================== 几何参数 ====================
    # 平板层定义 (name, z_start, thickness, color, opacity)
    layers = [
        ('skin', 0.0, 0.2, 'sandybrown', 0.6),      # 皮肤 - 浅棕色
        ('skull', 0.2, 0.8, 'ivory', 0.8),          # 颅骨 - 象牙白
        ('brain', 1.0, 16.0, 'lightpink', 0.4),     # 脑组织 - 浅粉色，半透明
    ]
    
    # 球体定义 (name, center, radius, color)
    # 使用当前配置 (0, 0, 5)
    spheres = [
        ('hematoma', [0.0, 0.0, 5.0], 0.5, 'darkred'),
    ]
    
    # 源和探测器
    source_pos = [0.0, 0.0, -1.0]
    detector_z = 17.5
    
    # ==================== 绘制平板层 ====================
    print("Adding layers...")
    for name, z_start, thickness, color, opacity in layers:
        mesh = create_layer_mesh(z_start, thickness, color=color)
        plotter.add_mesh(mesh, color=color, opacity=opacity, 
                        show_edges=True, edge_color='black', 
                        line_width=0.5, label=name)
        
        # 添加层标签
        label_pos = [8, 8, z_start + thickness/2]
        plotter.add_point_labels([label_pos], [f'{name}\n({thickness}cm)'], 
                                font_size=10, text_color='black',
                                point_size=0, shape=None)
    
    # ==================== 绘制球体 ====================
    print("Adding spheres...")
    for name, center, radius, color in spheres:
        mesh = create_sphere_mesh(center, radius, color=color)
        plotter.add_mesh(mesh, color=color, opacity=0.8,
                        show_edges=True, edge_color='darkred',
                        line_width=2, label=name)
        
        # 添加球体标签
        label_pos = [center[0] + radius + 1, center[1], center[2]]
        plotter.add_point_labels([label_pos], 
                                [f'{name}\nr={radius}cm'], 
                                font_size=10, text_color='darkred',
                                point_size=0, shape=None)
        
        # 添加球体虚线框（显示边界）
        wireframe = pv.Sphere(radius=radius, center=center, 
                             theta_resolution=20, phi_resolution=20)
        plotter.add_mesh(wireframe, color='darkred', opacity=0.3,
                        style='wireframe', line_width=1)
    
    # ==================== 绘制探测器 ====================
    print("Adding detector...")
    detector = create_detector_mesh(detector_z, color='lightblue')
    plotter.add_mesh(detector, color='lightblue', opacity=0.5,
                    show_edges=True, edge_color='blue',
                    line_width=2, label='detector')
    
    # 探测器标签
    plotter.add_point_labels([[8, 8, detector_z]], 
                            ['Detector\n(1024×1024)'], 
                            font_size=10, text_color='blue',
                            point_size=0, shape=None)
    
    # ==================== 绘制光源 ====================
    print("Adding source...")
    source = create_source_marker(source_pos, color='gold')
    plotter.add_mesh(source, color='gold', opacity=1.0,
                    show_edges=True, edge_color='orange',
                    line_width=2, label='source')
    
    # 光源标签
    plotter.add_point_labels([[source_pos[0] + 2, source_pos[1], source_pos[2]]], 
                            ['X-ray Source'], 
                            font_size=10, text_color='orange',
                            point_size=0, shape=None)
    
    # ==================== 绘制光子射线示例 ====================
    print("Adding sample photon rays...")
    photon_rays = create_photon_rays(source_pos, num_rays=15, spread=2.0)
    for i, ray in enumerate(photon_rays):
        # 交替颜色
        color = 'green' if i % 2 == 0 else 'lime'
        plotter.add_mesh(ray, color=color, line_width=1.5, opacity=0.6)
    
    # ==================== 添加坐标轴标签 ====================
    # 添加Z轴方向的箭头表示光子传播方向
    arrow_start = [0, -12, 0]
    arrow_end = [0, -12, 10]
    direction = np.array(arrow_end) - np.array(arrow_start)
    arrow = pv.Arrow(start=arrow_start, direction=direction, shaft_radius=0.2, 
                     tip_radius=0.4, tip_length=2)
    plotter.add_mesh(arrow, color='green')
    plotter.add_point_labels([[arrow_start[0], arrow_start[1] - 2, arrow_start[2] + 5]], 
                            ['Photon\nDirection'], 
                            font_size=9, text_color='green',
                            point_size=0, shape=None)
    
    # ==================== 设置视角和标题 ====================
    plotter.camera_position = [
        (35, 35, 10),   # 相机位置
        (0, 0, 8),      # 焦点
        (0, 0, 1)       # 上方向
    ]
    
    plotter.add_title('Medical Imaging Photon Transport Simulation\n'
                     'Layered Head Model with Sphere Inclusion',
                     font_size=14, color='black')
    
    # 添加图例
    plotter.add_legend(face='circle', size=(0.1, 0.1))
    
    # 添加尺寸标注
    # Z轴尺寸线
    plotter.add_mesh(pv.Line([12, -10, 0], [12, -10, 17.5]), 
                    color='gray', line_width=2)
    plotter.add_point_labels([[12, -10, 8.75]], ['17.5 cm'], 
                            font_size=9, text_color='gray',
                            point_size=0, shape=None)
    
    # ==================== 保存和显示 ====================
    # print("Saving screenshot...")
    # plotter.screenshot('../output/geometry_3d_view.png',
    #                   window_size=[1200, 900])
    # print("Saved: output/geometry_3d_view.png")

    # 交互式显示（可选）
    print("Launching interactive viewer...")
    print("Controls: Left-click rotate, Right-click zoom, Middle-click pan")
    print("Press 'q' to quit")
    plotter.show()


def create_comparison_view():
    """创建对比视图：无球体 vs 有球体"""
    
    plotter = pv.Plotter(shape=(1, 2))
    
    # 左图：无球体
    plotter.subplot(0, 0)
    plotter.set_background('white')
    plotter.add_title('Without Sphere', font_size=12)
    
    # 绘制层
    layers = [
        ('skin', 0.0, 0.2, 'sandybrown', 0.6),
        ('skull', 0.2, 0.8, 'ivory', 0.8),
        ('brain', 1.0, 16.0, 'lightpink', 0.4),
    ]
    for name, z_start, thickness, color, opacity in layers:
        mesh = create_layer_mesh(z_start, thickness, color=color)
        plotter.add_mesh(mesh, color=color, opacity=opacity, 
                        show_edges=True, edge_color='black')
    
    # 源和探测器
    source = create_source_marker([0, 0, -1], color='gold')
    plotter.add_mesh(source, color='gold')
    detector = create_detector_mesh(17.5, color='lightblue')
    plotter.add_mesh(detector, color='lightblue', opacity=0.5)
    
    plotter.camera_position = [(30, 30, 10), (0, 0, 8), (0, 0, 1)]
    
    # 右图：有球体
    plotter.subplot(0, 1)
    plotter.set_background('white')
    plotter.add_title('With Sphere (Hematoma)', font_size=12)
    
    # 绘制层
    for name, z_start, thickness, color, opacity in layers:
        mesh = create_layer_mesh(z_start, thickness, color=color)
        plotter.add_mesh(mesh, color=color, opacity=opacity, 
                        show_edges=True, edge_color='black')
    
    # 球体
    sphere = create_sphere_mesh([0, 0, 5], 0.5, color='darkred')
    plotter.add_mesh(sphere, color='darkred', opacity=0.8)
    
    # 源和探测器
    source = create_source_marker([0, 0, -1], color='gold')
    plotter.add_mesh(source, color='gold')
    detector = create_detector_mesh(17.5, color='lightblue')
    plotter.add_mesh(detector, color='lightblue', opacity=0.5)
    
    plotter.camera_position = [(30, 30, 10), (0, 0, 8), (0, 0, 1)]
    
    # 保存
    plotter.screenshot('../output/geometry_comparison.png', 
                      window_size=[1600, 800])
    print("Saved: output/geometry_comparison.png")
    
    plotter.show()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--compare':
        create_comparison_view()
    else:
        main()
