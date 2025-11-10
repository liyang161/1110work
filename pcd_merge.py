import os
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

# --------------------------
# 配置区域：修改此处适应你的数据
# --------------------------
# 1. 四组点云的位姿字典（必填）
# 格式：{点云文件名: [x, y, z, rx, ry, rz]}
# 单位：x/y/z为米(m)，rx/ry/rz为旋转矢量（弧度）
point_cloud_poses = {
    "out\\x=-43.75\\scan_0001.txt": [-0.04375, -0.16451, 0.79252, 0.884, 2.629, -0.264],
    "out\\x=-61.26\\scan_0001.txt": [-0.06126, -0.11841, 0.73504, 0.692, 2.617, -0.429],
    "out\\x=-69.29\\scan_0001.txt": [-0.06929, -0.04890, 0.55885, 1.003, 2.105, -0.293],
    "out\\x=-146.77\\scan_0001.txt": [-0.14677, -0.17205, 0.56815, 2.180, -0.579, 1.668]
}
target_name = "out\\x=-43.75\\scan_0001.txt"
# 2. 点云文件存放目录（必填）
PCD_DIRECTORY = "./point_clouds"  # 示例：所有TXT点云放在该文件夹下

# 3. 配准参数（可选，根据点云精度调整）
VOXEL_SIZE = 5.0  # 降采样体素大小（单位：毫米）
ICP_MAX_DISTANCE = 10.0  # ICP最大对应点距离（单位：毫米）

def copy_point_cloud(pcd):
    """手动复制点云数据，替代clone()方法"""
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
    if pcd.has_colors():
        new_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
    if pcd.has_normals():
        new_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals))
    return new_pcd
# --------------------------
# 核心功能函数
# --------------------------
def load_txt_point_cloud(filename):
    """加载TXT格式点云（x,y,z坐标，支持空格/逗号分隔）"""
    # file_path = os.path.join(PCD_DIRECTORY, filename)
    file_path = filename
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"点云文件不存在: {file_path}")

    # 尝试读取文件（支持多种分隔符）
    try:
        # 优先尝试空格分隔
        points = np.loadtxt(file_path, dtype=np.float64)
    except:
        # 失败则尝试逗号分隔
        points = np.loadtxt(file_path, dtype=np.float64, delimiter=",")

    # 验证点云格式
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"点云格式错误，需至少3列(x,y,z)，实际为{points.shape[1]}列")

    # 提取x,y,z坐标（忽略额外列）
    points = points[:, :3]

    # 单位转换（米→毫米）
    if np.max(np.abs(points)) < 10:  # 经验判断：坐标值较小则单位为米
        points *= 1000
        print(f"提示：{filename} 已从米转换为毫米")

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    print(f"加载点云: {filename}（{len(points)}个点）")
    return pcd


def get_pose(filename):
    """从字典获取点云位姿并转换单位（米→毫米）"""
    if filename not in point_cloud_poses:
        raise KeyError(f"位姿字典中未找到 {filename} 的位姿信息")

    pose = point_cloud_poses[filename].copy()
    # x,y,z单位转换（米→毫米）
    pose[0] *= 1000
    pose[1] *= 1000
    pose[2] *= 1000
    return pose


def pose_to_matrix(pose):
    """将位姿[x,y,z,rx,ry,rz]转换为4x4变换矩阵"""
    x, y, z, rx, ry, rz = pose
    # 旋转矢量→旋转矩阵
    rot_mat = R.from_rotvec([rx, ry, rz]).as_matrix()
    # 构造齐次变换矩阵
    trans_mat = np.eye(4)
    trans_mat[:3, :3] = rot_mat
    trans_mat[:3, 3] = [x, y, z]
    return trans_mat


def compute_relative_transform(pose_source, pose_target):
    """计算源点云到目标点云的相对变换矩阵"""
    T_source = pose_to_matrix(pose_source)  # 源点云→世界坐标系
    T_target = pose_to_matrix(pose_target)  # 目标点云→世界坐标系
    return np.linalg.inv(T_target) @ T_source  # 源→目标的变换矩阵

#
def coarse_registration(source, target):
    """基于FPFH特征的粗配准（修复参数顺序错误）"""
    # 降采样
    source_down = source.voxel_down_sample(VOXEL_SIZE)
    target_down = target.voxel_down_sample(VOXEL_SIZE)
    print(f"粗配准降采样：源点云{len(source_down.points)}点，目标点云{len(target_down.points)}点")

    # 计算法向量
    source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 2, max_nn=30)
    )
    target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 2, max_nn=30)
    )

    # 计算FPFH特征
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 5, max_nn=100)
    )
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 5, max_nn=100)
    )

    # RANSAC配准（修复参数顺序：添加mutual_filter参数）
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        mutual_filter=True,  # 新增：启用双向过滤（关键修复）
        max_correspondence_distance=VOXEL_SIZE * 1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(VOXEL_SIZE * 1.5)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )

    print(f"粗配准：重叠率={result.fitness:.4f}，RMSE={result.inlier_rmse:.4f}mm")
    return result


def fine_registration(source, target, init_transform):
    """基于ICP的精配准（优化初始变换）"""
    # 降采样（仅用于计算，不影响最终结果）
    source_down = source.voxel_down_sample(VOXEL_SIZE / 2)
    target_down = target.voxel_down_sample(VOXEL_SIZE / 2)

    # 计算法向量（用于点到面ICP）
    source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE, max_nn=30)
    )
    target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE, max_nn=30)
    )
    # 统一法向量方向（以10个邻点为参考）
    source_down.orient_normals_consistent_tangent_plane(10)
    target_down.orient_normals_consistent_tangent_plane(10)
    # ICP精配准
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down,
        max_correspondence_distance=ICP_MAX_DISTANCE,
        init=init_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=100,
            relative_fitness=1e-6,
            relative_rmse=1e-6
        )
    )
    print(f"精配准：重叠率={result.fitness:.4f}，RMSE={result.inlier_rmse:.4f}mm")
    return result


def visualize_result(source, target, transform, title):
    """可视化配准结果"""
    # 复制点云并着色
    source_copy = copy_point_cloud(source)
    target_copy = copy_point_cloud(target)
    source_copy.paint_uniform_color([1, 0.706, 0])  # 源点云：黄色
    target_copy.paint_uniform_color([0, 0.651, 0.929])  # 目标点云：蓝色

    # 应用变换
    source_copy.transform(transform)

    # 添加坐标系
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)  # 50mm坐标系

    # 可视化
    o3d.visualization.draw_geometries(
        [source_copy, target_copy, coord],
        window_name=title,
        width=1280,
        height=960,
        zoom=0.4,
        front=[0.9, -0.2, -0.3],
        lookat=[0, 0, 0],
        up=[-0.3, -0.9, -0.2]
    )


def merge_all_point_clouds(target_name):
    """融合所有点云到目标点云坐标系"""
    # 加载目标点云
    target_pcd = load_txt_point_cloud(target_name)
    target_pose = get_pose(target_name)
    merged_pcd = copy_point_cloud(target_pcd)
    merged_pcd.paint_uniform_color([0, 0.651, 0.929])  # 目标点云：蓝色
    init_merge = target_pcd
    fpfh_merge = target_pcd
    # 遍历所有源点云
    for source_name in point_cloud_poses.keys():
        if source_name == target_name:
            print("=====")
            continue  # 跳过目标点云

        # 加载源点云
        source_pcd = load_txt_point_cloud(source_name)
        source_pose = get_pose(source_name)
        print(source_pose)
        # 步骤1：位姿初始配准
        pose_transform = compute_relative_transform(source_pose, target_pose)
        source_init = source_pcd.transform(pose_transform)
        init_merge=init_merge+source_init
        # # 可视化最终融合结果
        # target_pcd.paint_uniform_color([0, 0.651, 0.929])  # 目标点云：蓝色
        # o3d.visualization.draw_geometries(
        #     [target_pcd,source_init,o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)],
        #     window_name="位姿初步配准融合结果"
        # )
        # 1. 对初始对齐后的source_init做粗配准（获取微调变换）
        fpfh_result = coarse_registration(source_init, target_pcd)
        # 2. 仅对source_init应用粗配准的微调变换（而非对原始source_pcd叠加变换）
        source_fpfh_aligned = copy_point_cloud(source_init)
        source_fpfh_aligned.transform(fpfh_result.transformation)

        # # 可视化对比：初始对齐 vs 粗配准后
        target_pcd.paint_uniform_color([0, 0.651, 0.929])  # 蓝色：目标
        source_init.paint_uniform_color([1, 0.706, 0])  # 黄色：初始对齐
        source_fpfh_aligned.paint_uniform_color([1, 0, 0])  # 红色：粗配准后
        o3d.visualization.draw_geometries(
            [target_pcd, source_init, source_fpfh_aligned, o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)],
            window_name="初始对齐 vs FPFH粗配准后"
        )
        fpfh_merge=fpfh_merge+source_fpfh_aligned
        # fpfh_result=coarse_registration(source_init, target_pcd)
        # source_transformed = source_pcd.transform(fpfh_result.transformation)
        # source_transformed.paint_uniform_color([1, 0.706, 0])  # 源点云：黄色
        # o3d.visualization.draw_geometries(
        #     [ target_pcd,source_transformed,o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)],
        #     window_name="fpfh粗配准融合结果"
        # )
        # # 步骤2：ICP精配准
        # icp_result_init = fine_registration(source_init, target_pcd, np.eye(4))
        # # final_transform = pose_transform @ icp_result.transformation
        #
        # # 应用最终变换并融合
        # source_transformed_init = source_pcd.transform(icp_result_init.transformation)
        # source_transformed_init.paint_uniform_color([1, 0.706, 0])  # 源点云：黄色
        # # merged_pcd += source_transformed_init
        # o3d.visualization.draw_geometries(
        #     [target_pcd,source_transformed_init,o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)],
        #     window_name="初始配准点云精融合结果"
        # )
        # 步骤2：ICP精配准
        icp_result_fpfh = fine_registration(source_fpfh_aligned, target_pcd, np.eye(4))

        # 应用最终变换并融合
        source_transformed_fpfh = source_fpfh_aligned.transform(icp_result_fpfh.transformation)
        # source_transformed_fpfh.paint_uniform_color([1, 0.706, 0])  # 源点云：黄色
        # # merged_pcd += source_transformed_init
        # o3d.visualization.draw_geometries(
        #     [target_pcd,source_transformed_fpfh,o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)],
        #     window_name="fpfh配准点云精融合结果"
        # )
        merged_pcd += source_transformed_fpfh
        # # 可视化当前配准结果
        # visualize_result(
        #     source_pcd, target_pcd, source_transformed,
        #     title=f"配准 {source_name} → {target_name}"
        # )

    # 保存融合结果
    # save_path = os.path.join(PCD_DIRECTORY, "merged_all_clouds.ply")
    # o3d.io.write_point_cloud(save_path, merged_pcd)
    # print(f"\n所有点云融合完成，保存至：{save_path}")
    # o3d.visualization.draw_geometries(
    #     [init_merge,o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)],
    #     window_name="初始融合结果"
    # )
    # o3d.visualization.draw_geometries(
    #     [fpfh_merge,o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)],
    #     window_name="fpfh配准点云融合结果"
    # )
    # o3d.visualization.draw_geometries(
    #     [target_pcd,o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)],
    #     window_name="fpfh配准点云精融合结果"
    # )
    txt_path = os.path.join(PCD_DIRECTORY, "merged_all_clouds_init.txt")
    points = np.asarray(init_merge.points)  # 只取坐标
    np.savetxt(txt_path, points, fmt='%.6f')  # 每行 “x y z”
    txt_path = os.path.join(PCD_DIRECTORY, "merged_all_clouds_fhph.txt")
    points = np.asarray(fpfh_merge.points)  # 只取坐标
    np.savetxt(txt_path, points, fmt='%.6f')  # 每行 “x y z”
    txt_path = os.path.join(PCD_DIRECTORY, "merged_all_clouds_icp.txt")
    points = np.asarray(merged_pcd.points)  # 只取坐标
    np.savetxt(txt_path, points, fmt='%.6f')  # 每行 “x y z”
    # # 可视化最终融合结果
    # o3d.visualization.draw_geometries(
    #     [merged_pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)],
    #     window_name="所有点云融合结果"
    # )


# --------------------------
# 主函数
# --------------------------
def main():
    # 检查点云目录是否存在
    if not os.path.exists(PCD_DIRECTORY):
        os.makedirs(PCD_DIRECTORY)
        print(f"已创建点云目录：{PCD_DIRECTORY}，请将TXT点云放入该目录")
        return

    # 融合所有点云（以第一组为目标）
    merge_all_point_clouds(target_name)


if __name__ == "__main__":
    main()
