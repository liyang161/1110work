import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import os
import glob
from pathlib import Path


class TurntableCoordinateSystem:
    def __init__(self, rotation_center, axis_direction, scan_origin_pose):
        """
        初始化转台坐标系系统
        rotation_center: 旋转中心点 [x, y, z]
        axis_direction: 转轴方向向量 [vx, vy, vz]
        scan_origin_pose: 扫描原点位姿 [x, y, z, rx, ry, rz] (位置+欧拉角)
        """
        self.rotation_center = np.array(rotation_center)
        self.axis_direction = np.array(axis_direction)
        self.scan_origin_pose = np.array(scan_origin_pose)

        # 归一化转轴方向
        self.axis_direction = self.axis_direction / np.linalg.norm(self.axis_direction)

        # 构建转台坐标系
        self.build_turntable_coordinate_system()

    def build_turntable_coordinate_system(self):
        """构建转台坐标系到基坐标系的变换矩阵"""
        # 转轴方向作为Z轴
        z_axis = self.axis_direction

        # 构建X轴（与Z轴垂直的任意方向）
        if abs(z_axis[0]) > 0.1:
            x_axis = np.cross(z_axis, np.array([0, 1, 0]))
        else:
            x_axis = np.cross(z_axis, np.array([1, 0, 0]))
        x_axis = x_axis / np.linalg.norm(x_axis)

        # 构建Y轴（通过叉积）
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # 构建旋转矩阵
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

        # 构建齐次变换矩阵：从转台坐标系到基坐标系
        self.T_turntable_to_base = np.eye(4)
        self.T_turntable_to_base[:3, :3] = rotation_matrix
        self.T_turntable_to_base[:3, 3] = self.rotation_center

        # 构建基坐标系到转台坐标系的逆变换
        self.T_base_to_turntable = np.linalg.inv(self.T_turntable_to_base)

        print("转台坐标系到基坐标系的变换矩阵:")
        print(self.T_turntable_to_base)

    def get_rotation_transform(self, angle_degrees):
        """
        根据转台旋转角度获取变换矩阵
        angle_degrees: 旋转角度（度）
        """
        # 将角度转换为弧度
        angle_rad = np.radians(angle_degrees)

        # 创建绕转轴旋转的变换矩阵
        rotation_transform = np.eye(4)

        # 使用罗德里格斯公式计算旋转矩阵
        k = self.axis_direction
        K = np.array([[0, -k[2], k[1]],
                      [k[2], 0, -k[0]],
                      [-k[1], k[0], 0]])

        rotation_matrix = (np.eye(3) + np.sin(angle_rad) * K +
                           (1 - np.cos(angle_rad)) * np.dot(K, K))

        rotation_transform[:3, :3] = rotation_matrix
        rotation_transform[:3, 3] = (self.rotation_center -
                                     np.dot(rotation_matrix, self.rotation_center))

        return rotation_transform

    def parse_angle_from_filename(self, filename):
        """
        从文件名中解析旋转角度
        假设文件名格式如: scan_30deg.ply, pointcloud_45.pcd等
        """
        filename = Path(filename).stem.lower()

        # 多种角度解析策略
        import re

        # 匹配数字模式
        patterns = [
            r'(\d+)deg',  # 30deg
            r'angle[_-]?(\d+)',  # angle30, angle_30
            r'(\d+)',  # 纯数字
            r'rot[_-]?(\d+)'  # rot30, rot_30
        ]

        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return int(match.group(1))

        # 如果无法解析，返回0度（默认）
        print(f"警告: 无法从文件名 {filename} 解析角度，使用默认角度0度")
        return 0


class PointCloudStitcher:
    def __init__(self, turntable_system):
        self.turntable_system = turntable_system
        self.pointclouds = []
        self.transforms = []

    def load_pointclouds_from_folder(self, folder_path, file_extension=".ply"):
        """从文件夹加载所有点云文件"""
        folder_path = Path(folder_path)
        pointcloud_files = glob.glob(str(folder_path / f"*{file_extension}"))

        if not pointcloud_files:
            raise FileNotFoundError(f"在文件夹 {folder_path} 中未找到 {file_extension} 文件")

        print(f"找到 {len(pointcloud_files)} 个点云文件")

        for file_path in pointcloud_files:
            try:
                # 加载点云
                pcd = o3d.io.read_point_cloud(file_path)

                if len(pcd.points) == 0:
                    print(f"警告: 文件 {file_path} 为空，跳过")
                    continue

                # 从文件名解析旋转角度
                angle = self.turntable_system.parse_angle_from_filename(file_path)

                # 存储点云和对应的角度
                self.pointclouds.append({
                    'pointcloud': pcd,
                    'file_path': file_path,
                    'angle': angle,
                    'transform': self.turntable_system.get_rotation_transform(angle)
                })

                print(f"加载点云: {Path(file_path).name}, 角度: {angle}度")

            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {e}")

        return len(self.pointclouds)

    def apply_transforms(self):
        """应用变换矩阵到所有点云"""
        transformed_pointclouds = []

        for i, data in enumerate(self.pointclouds):
            pcd = data['pointcloud']
            transform = data['transform']

            # 应用变换
            pcd_transformed = pcd.transform(transform)
            transformed_pointclouds.append(pcd_transformed)

            print(f"应用变换到点云 {i + 1}, 角度: {data['angle']}度")

        return transformed_pointclouds

    def merge_pointclouds(self, voxel_size=0.001):
        """合并所有变换后的点云"""
        if not self.pointclouds:
            raise ValueError("没有可合并的点云")

        transformed_pcds = self.apply_transforms()

        # 合并点云
        merged_pcd = transformed_pcds[0]

        for i in range(1, len(transformed_pcds)):
            merged_pcd += transformed_pcds[i]

        # 体素下采样以减少点云密度（可选）
        if voxel_size > 0:
            merged_pcd = merged_pcd.voxel_down_sample(voxel_size=voxel_size)

        print(f"合并完成，总点数: {len(merged_pcd.points)}")

        return merged_pcd

    def visualize_individual(self):
        """分别显示每个点云（不同颜色）"""
        if not self.pointclouds:
            print("没有可显示的点云")
            return

        transformed_pcds = self.apply_transforms()

        # 为每个点云分配不同颜色
        colors = [
            [1, 0, 0],  # 红色
            [0, 1, 0],  # 绿色
            [0, 0, 1],  # 蓝色
            [1, 1, 0],  # 黄色
            [1, 0, 1],  # 紫色
            [0, 1, 1],  # 青色
            [1, 0.5, 0],  # 橙色
            [0.5, 0, 1]  # 紫色
        ]

        colored_pcds = []
        for i, pcd in enumerate(transformed_pcds):
            color = colors[i % len(colors)]
            pcd.paint_uniform_color(color)
            colored_pcds.append(pcd)

        # 创建坐标系
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=50, origin=[0, 0, 0])

        # 显示
        o3d.visualization.draw_geometries(colored_pcds + [coordinate_frame],
                                          window_name="多视角点云显示")

    def visualize_merged(self, voxel_size=0.001):
        """显示合并后的点云"""
        merged_pcd = self.merge_pointclouds(voxel_size=voxel_size)

        # 设置为灰色
        merged_pcd.paint_uniform_color([0.5, 0.5, 0.5])

        # 创建坐标系
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=50, origin=[0, 0, 0])

        # 显示
        o3d.visualization.draw_geometries([merged_pcd, coordinate_frame],
                                          window_name="合并点云显示")

    def save_merged_pointcloud(self, output_path, voxel_size=0.001):
        """保存合并后的点云"""
        merged_pcd = self.merge_pointclouds(voxel_size=voxel_size)
        o3d.io.write_point_cloud(output_path, merged_pcd)
        print(f"合并点云已保存到: {output_path}")


def main():
    # 1. 初始化转台坐标系参数（使用您提供的实际数据）
    rotation_center = [-104.047, 5.721, 567.681]  # 圆心坐标
    axis_direction = [0.938301, -0.033760, 0.344168]  # 法向量方向
    scan_origin_pose = [-113.06, -4.91, 481.07, 0.672, 1.791, -0.526]  # 扫描原点位姿

    # 2. 创建转台坐标系实例
    turntable_system = TurntableCoordinateSystem(
        rotation_center, axis_direction, scan_origin_pose
    )

    # 3. 创建点云拼接器
    stitcher = PointCloudStitcher(turntable_system)

    # 4. 指定点云文件夹路径
    pointcloud_folder = "1107\\ply"  # 请修改为实际路径

    try:
        # 5. 加载点云文件
        num_loaded = stitcher.load_pointclouds_from_folder(
            pointcloud_folder, file_extension=".ply"  # 根据实际文件类型修改
        )

        if num_loaded == 0:
            print("没有成功加载任何点云文件")
            return

        print(f"成功加载 {num_loaded} 个点云文件")

        # 6. 显示加载信息
        print("\n=== 点云加载信息 ===")
        for i, data in enumerate(stitcher.pointclouds):
            print(f"点云 {i + 1}: 角度={data['angle']}度, 文件={Path(data['file_path']).name}")

        # 7. 可视化选项
        while True:
            print("\n=== 可视化选项 ===")
            print("1. 分别显示各视角点云（不同颜色）")
            print("2. 显示合并后的点云")
            print("3. 保存合并点云")
            print("4. 退出")

            choice = input("请选择操作 (1-4): ").strip()

            if choice == "1":
                print("正在显示多视角点云...")
                stitcher.visualize_individual()

            elif choice == "2":
                print("正在合并并显示点云...")
                stitcher.visualize_merged(voxel_size=0.001)

            elif choice == "3":
                output_path = "merged_pointcloud.ply"
                stitcher.save_merged_pointcloud(output_path)
                print(f"点云已保存到: {output_path}")

            elif choice == "4":
                print("程序退出")
                break
            else:
                print("无效选择，请重新输入")

    except Exception as e:
        print(f"程序执行出错: {e}")


# 简化使用函数
def quick_stitch(pointcloud_folder, output_file="merged_result.ply"):
    """
    快速拼接函数
    pointcloud_folder: 点云文件夹路径
    output_file: 输出文件名
    """
    # 使用您的实际标定参数
    rotation_center = [-104.047, 5.721, 567.681]
    axis_direction = [0.938301, -0.033760, 0.344168]
    scan_origin_pose = [-113.06, -4.91, 481.07, 0.672, 1.791, -0.526]

    turntable_system = TurntableCoordinateSystem(
        rotation_center, axis_direction, scan_origin_pose
    )

    stitcher = PointCloudStitcher(turntable_system)

    # 加载并处理点云
    stitcher.load_pointclouds_from_folder(pointcloud_folder)
    stitcher.save_merged_pointcloud(output_file)
    stitcher.visualize_merged()

    print("点云拼接完成!")


if __name__ == "__main__":
    # 直接运行主程序
    main()

    # 或者使用快速拼接
    # quick_stitch("path/to/your/pointclouds")