import numpy as np
from scipy.optimize import least_squares
import open3d as o3d


def fit_circle_3d(points):
    """
    在三维空间中拟合圆

    参数:
    points: numpy数组, 形状为(6,3)，包含6个三维点坐标

    返回:
    center: 圆的中心点(旋转中心)
    normal: 圆的法向量(转轴方向)
    radius: 圆的半径
    """

    # 步骤1: 计算质心
    centroid = np.mean(points, axis=0)

    # 步骤2: 计算协方差矩阵并找到最佳拟合平面
    centered_points = points - centroid
    covariance_matrix = centered_points.T @ centered_points

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # 最小特征值对应的特征向量是平面的法向量
    normal = eigenvectors[:, 0]
    normal = normal / np.linalg.norm(normal)  # 归一化

    # 步骤3: 将点投影到拟合平面上
    # 计算每个点到平面的距离
    distances = np.dot(centered_points, normal)

    # 将点投影到平面上
    projected_points = points - np.outer(distances, normal)

    # 步骤4: 在投影点上拟合二维圆
    # 建立平面坐标系
    # 选择两个正交向量作为平面基
    if np.abs(normal[2]) > 0.5:
        u_vec = np.array([1, 0, 0])
    else:
        u_vec = np.array([0, 0, 1])

    u_vec = u_vec - np.dot(u_vec, normal) * normal
    u_vec = u_vec / np.linalg.norm(u_vec)
    v_vec = np.cross(normal, u_vec)

    # 将投影点转换到平面坐标系
    plane_coords = np.column_stack([
        np.dot(projected_points - centroid, u_vec),
        np.dot(projected_points - centroid, v_vec)
    ])

    # 使用最小二乘法拟合二维圆
    def circle_residuals(params, points_2d):
        cx, cy, r = params
        residuals = np.sqrt((points_2d[:, 0] - cx) ** 2 + (points_2d[:, 1] - cy) ** 2) - r
        return residuals

    # 初始猜测：质心为圆心，平均距离为半径
    initial_guess = [0, 0, np.mean(np.linalg.norm(plane_coords, axis=1))]

    # 使用最小二乘法优化
    result = least_squares(circle_residuals, initial_guess, args=(plane_coords,))
    cx, cy, radius = result.x

    # 将二维圆心坐标转换回三维空间
    center_2d = np.array([cx, cy])
    center_3d = centroid + cx * u_vec + cy * v_vec

    return center_3d, normal, radius


def create_coordinate_frame(size=1.0, origin=(0, 0, 0)):
    """创建坐标系框架"""
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
    return frame


def create_arrow(origin, direction, length, color):
    """创建箭头"""
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=length * 0.02,
        cone_radius=length * 0.04,
        cylinder_height=length * 0.8,
        cone_height=length * 0.2
    )
    arrow.compute_vertex_normals()

    # 旋转箭头使其指向正确方向
    z_axis = np.array([0, 0, 1])
    rot_axis = np.cross(z_axis, direction)
    rot_angle = np.arccos(np.dot(z_axis, direction) / (np.linalg.norm(z_axis) * np.linalg.norm(direction)))
    if np.linalg.norm(rot_axis) > 1e-6:
        rot_axis = rot_axis / np.linalg.norm(rot_axis)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * rot_angle)
        arrow.rotate(R, center=(0, 0, 0))

    # 平移箭头到正确位置
    arrow.translate(origin)
    arrow.paint_uniform_color(color)
    return arrow


def create_circle(center, normal, radius, color, resolution=100):
    """创建圆环"""
    # 建立平面坐标系
    if np.abs(normal[2]) > 0.5:
        u_vec = np.array([1, 0, 0])
    else:
        u_vec = np.array([0, 0, 1])

    u_vec = u_vec - np.dot(u_vec, normal) * normal
    u_vec = u_vec / np.linalg.norm(u_vec)
    v_vec = np.cross(normal, u_vec)

    # 生成圆上的点
    theta = np.linspace(0, 2 * np.pi, resolution)
    circle_points = np.array([
        center + radius * (np.cos(angle) * u_vec + np.sin(angle) * v_vec)
        for angle in theta
    ])

    # 创建线集
    lines = [[i, (i + 1) % resolution] for i in range(resolution)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(circle_points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.paint_uniform_color(color)
    return line_set


# 使用提供的6个标定球中心点
points = np.array([
    [-80.289, -5.996, 501.672],  # 点1
    [-97.510, -61.248, 543.379],  # 点2
    [-121.273, -49.611, 609.183],  # 点3
    [-127.823, 17.326, 633.626],  # 点4
    [-110.646, 72.664, 592.252],  # 点5
    [-86.864, 61.070, 526.306]  # 点6
])

print("输入点坐标:")
for i, point in enumerate(points):
    print(f"点{i + 1}: ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})")

# 拟合圆
center, normal, radius = fit_circle_3d(points)

print(f"\n拟合结果:")
print(f"旋转中心(圆心): ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
print(f"转轴方向(法向量): ({normal[0]:.6f}, {normal[1]:.6f}, {normal[2]:.6f})")
print(f"半径: {radius:.3f}")

# 创建Open3D可视化
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='空间圆拟合与转轴计算', width=800, height=600)

# 添加坐标系框架 (放大显示)
coord_frame = create_coordinate_frame(size=20.0, origin=(0, 0, 0))
vis.add_geometry(coord_frame)

# 添加原始点云
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.paint_uniform_color([1, 0, 0])  # 红色
vis.add_geometry(pcd)

# 添加旋转中心
center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
center_sphere.translate(center)
center_sphere.paint_uniform_color([0, 0, 1])  # 蓝色
vis.add_geometry(center_sphere)

# 添加转轴 (放大显示)
axis_length = 30.0
axis_arrow = create_arrow(center - normal * axis_length / 2, normal, axis_length, [0, 1, 0])  # 绿色
vis.add_geometry(axis_arrow)

# 添加拟合圆
circle = create_circle(center, normal, radius, [0, 0, 1])  # 蓝色
vis.add_geometry(circle)

# 设置相机视角
ctr = vis.get_view_control()
ctr.set_front([0, 0, -1])  # 从正前方看
ctr.set_up([0, 1, 0])  # Y轴朝上
ctr.set_zoom(0.5)

# 运行可视化
vis.run()
vis.destroy_window()

# 计算转轴与坐标轴的夹角
angles_deg = np.degrees(np.arccos(np.abs(normal)))
print(f"\n转轴与坐标轴夹角:")
print(f"与X轴夹角: {angles_deg[0]:.2f}°")
print(f"与Y轴夹角: {angles_deg[1]:.2f}°")
print(f"与Z轴夹角: {angles_deg[2]:.2f}°")

# 计算转轴的直线方程参数
print(f"\n转轴直线方程:")
print(f"x = {center[0]:.3f} + t * {normal[0]:.6f}")
print(f"y = {center[1]:.3f} + t * {normal[1]:.6f}")
print(f"z = {center[2]:.3f} + t * {normal[2]:.6f}")
print("(t为参数)")
