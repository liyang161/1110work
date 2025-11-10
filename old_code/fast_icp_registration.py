# -*- coding: utf-8 -*-
"""
配准改进版（针对配准结果不佳的情况）
- 增加可视化检查
- 参数可调（体素、RANSAC、ICP）
- 支持 Point‑to‑Point ICP（法向不可靠时使用）
- 可选离群点剔除
"""

import copy
import numpy as np
import open3d as o3d
import os


def load_txt_point_cloud(txt_path):
    """读取 txt（x y z 或 x y z nx ny nz）"""
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"文件不存在: {txt_path}")

    data = np.loadtxt(txt_path, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] not in (3, 6):
        raise ValueError("txt 文件每行应为 3 或 6 列 (x y z [nx ny nz])")

    points = data[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if data.shape[1] == 6:
        pcd.normals = o3d.utility.Vector3dVector(data[:, 3:6])
    return pcd


def preprocess_point_cloud(pcd, voxel_size, remove_outlier=False):
    """下采样 → 可选离群点剔除 → 法向估计 → FPFH"""
    pcd_down = pcd.voxel_down_sample(voxel_size)

    if remove_outlier:
        # 统计离群点剔除（可自行调参）
        pcd_down, ind = pcd_down.remove_statistical_outlier(nb_neighbors=20,
                                                            std_ratio=2.0)
        # 若想保留索引，可打印 len(ind) 查看被剔除比例

    if not pcd_down.has_normals():
        radius_normal = voxel_size * 3   # 放宽法向估计半径
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal,
                                                 max_nn=30)
        )
        # 法向统一指向外部（可选）
        pcd_down.orient_normals_consistent_tangent_plane(100)

    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature,
                                             max_nn=100),
    )
    return pcd_down, fpfh


def execute_fast_global_registration(source_down, target_down,
                                      source_fpfh, target_fpfh,
                                      voxel_size,
                                      distance_factor=1.5,
                                      ransac_n=4,
                                      max_iter=5000000,
                                      max_validation=1000):
    """RANSAC 粗配准（可调阈值、迭代次数）"""
    distance_threshold = voxel_size * distance_factor
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iter,
                                                                     max_validation),
    )
    return result


def refine_registration(source_down, target_down,
                        init_transform,
                        voxel_size,
                        use_point_to_point=False):
    """ICP 精配准（默认点到平面，可切换为点到点）"""
    distance_threshold = voxel_size * 0.4
    if use_point_to_point:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()

    result = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        distance_threshold,
        init_transform,
        estimation,
    )
    return result


def draw_clouds(clouds, titles=None):
    """统一的可视化函数，避免窗口尺寸警告"""
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)
    for i, pc in enumerate(clouds):
        vis.add_geometry(pc)
    if titles:
        for i, t in enumerate(titles):
            print(f"[{i}] {t}")
    vis.run()
    vis.destroy_window()


def draw_registration_result(source, target, transformation):
    """配准后可视化（使用原始点云）"""
    src_tmp = copy.deepcopy(source)
    src_tmp.transform(transformation)
    draw_clouds([src_tmp, target], ["配准后 source", "target"])


if __name__ == "__main__":
    # ------------------- 1. 读取 -------------------
    source = load_txt_point_cloud("shabiawei\\1.txt")
    target = load_txt_point_cloud("shabiawei\\2.txt")

    # ------------------- 2. 初步可视化（检查重叠） -------------------
    print("=== 初始点云可视化（检查是否有明显重叠） ===")
    draw_clouds([source, target], ["source (原始)", "target (原始)"])

    # ------------------- 3. 参数设置 -------------------
    voxel_size = 0.05          # 根据点云密度自行调节
    remove_outlier = True      # 是否进行离群点剔除
    use_point_to_point_icp = False   # 法向不可靠时设为 True

    # ------------------- 4. 预处理 -------------------
    source_down, source_fpfh = preprocess_point_cloud(source,
                                                      voxel_size,
                                                      remove_outlier)
    target_down, target_fpfh = preprocess_point_cloud(target,
                                                      voxel_size,
                                                      remove_outlier)

    # ------------------- 5. 粗配准 -------------------
    print("\nRunning fast global registration ...")
    result_fast = execute_fast_global_registration(
        source_down, target_down,
        source_fpfh, target_fpfh,
        voxel_size,
        distance_factor=2.0,   # 放宽匹配阈值（可调）
        ransac_n=4,
        max_iter=8000000,
        max_validation=1500
    )
    print("粗配准变换矩阵:\n", result_fast.transformation)

    # 可视化粗配准效果（下采样点云）
    src_tmp = copy.deepcopy(source_down)
    src_tmp.transform(result_fast.transformation)
    draw_clouds([src_tmp, target_down],
                ["source 粗配准后 (下采样)", "target (下采样)"])

    # ------------------- 6. 精配准 -------------------
    print("\nRefining registration with ICP ...")
    result_icp = refine_registration(source_down, target_down,
                                     result_fast.transformation,
                                     voxel_size,
                                     use_point_to_point=use_point_to_point_icp)
    print("精配准变换矩阵:\n", result_icp.transformation)

    # ------------------- 7. 最终可视化 -------------------
    draw_registration_result(source, target, result_icp.transformation)

    # ------------------- 8. 结果评估（可选） -------------------
    # 计算配准误差（均方根误差 RMSE）
    rmse = result_icp.inlier_rmse
    fitness = result_icp.fitness
    print(f"\n配准评估：RMSE = {rmse:.6f},  重叠度 (fitness) = {fitness:.4f}")
