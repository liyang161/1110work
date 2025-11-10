# import open3d as o3d
# import numpy as np
# import os
# import time
#
#
# def read_point_cloud(file_path):
#     """读取点云文件"""
#     start_time = time.time()
#     try:
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"文件不存在: {file_path}")
#
#         data = np.loadtxt(file_path, dtype=np.float32)
#         if data.size == 0 or data.ndim != 2 or data.shape[1] < 3:
#             raise ValueError(f"无效格式，需至少3列数据，实际{data.shape[1]}列")
#
#         points = data[:, :3]
#         print(f"原始点云: {len(points)} 个点（读取耗时: {time.time() - start_time:.2f}s）")
#
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(points)
#         return pcd
#
#     except Exception as e:
#         print(f"读取失败: {e}")
#         raise
#
#
# def downsample_point_cloud(pcd, voxel_size=0.01):
#     """降采样加速平面检测"""
#     start_time = time.time()
#     downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
#     print(f"降采样后: {len(downpcd.points)} 个点（耗时: {time.time() - start_time:.2f}s）")
#     return downpcd
#
#
# def find_largest_plane(pcd, distance_threshold=0.0015, ransac_n=5, num_iterations=500):
#     """检测最大平面"""
#     start_time = time.time()
#     plane_model, inliers = pcd.segment_plane(
#         distance_threshold=distance_threshold,
#         ransac_n=ransac_n,
#         num_iterations=num_iterations
#     )
#     a, b, c, d = plane_model
#     print(f"平面检测完成（耗时: {time.time() - start_time:.2f}s）")
#     print(f"平面方程: {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")
#     return plane_model
#
#
# def determine_far_side_from_origin(plane_model):
#     """确定平面距离原点较远的一侧"""
#     a, b, c, d = plane_model
#     # 平面法向量方向判断
#     return -np.sign(d)
#
#
# def filter_points(original_pcd, plane_model, far_side_sign, delete_plane_range=0.003):
#     """
#     过滤点云：
#     1. 删除平面两侧±delete_plane_range（3mm）范围内的点
#     2. 删除平面距离原点较远一侧的所有点
#     保留：距离原点较近一侧且距离平面>3mm的点
#     """
#     start_time = time.time()
#     a, b, c, d = plane_model
#     points = np.asarray(original_pcd.points)
#
#     # 计算符号距离（判断方向）和绝对距离（判断与平面的距离）
#     signed_dist = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
#     abs_dist = np.abs(signed_dist)
#
#     # 保留条件：
#     # 1. 不在远侧（符号与far_side_sign不同）
#     # 2. 距离平面超过3mm（不在平面删除范围内）
#     keep_mask = (np.sign(signed_dist) != far_side_sign) & (abs_dist > delete_plane_range)
#     filtered_points = points[keep_mask]
#
#     filtered_pcd = o3d.geometry.PointCloud()
#     filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
#
#     print(f"点云过滤完成（耗时: {time.time() - start_time:.2f}s）")
#     print(f"删除平面±{delete_plane_range * 1000}mm范围及远侧点，保留 {len(filtered_pcd.points)} 个点")
#     return filtered_pcd
#
#
# def visualize_result(filtered_pcd, vis_voxel_size=0.02):
#     """可视化处理结果"""
#     start_time = time.time()
#     vis_pcd = filtered_pcd.voxel_down_sample(voxel_size=vis_voxel_size)
#     vis_pcd.paint_uniform_color([0, 0.6, 1])
#
#     print(f"\n可视化准备完成（耗时: {time.time() - start_time:.2f}s）")
#     print("操作提示：鼠标拖动=旋转，滚轮=缩放，Shift+拖动=平移，Q=退出")
#
#     o3d.visualization.draw_geometries(
#         [vis_pcd],
#         window_name="点云处理结果（保留近侧且远离平面的点）",
#         width=1280,
#         height=960
#     )
#
#
# def save_filtered_point_cloud(pcd, output_path):
#     """保存过滤后的点云"""
#     start_time = time.time()
#     np.savetxt(output_path, np.asarray(pcd.points), fmt='%.6f')
#     print(f"结果保存完成（耗时: {time.time() - start_time:.2f}s）")
#
#
# def main():
#     total_start = time.time()
#
#     # 参数设置
#     input_file = "saomiao109\\x=-146.77\\scan_0001.txt"
#     output_file = "shabiawei\\4.txt"
#     delete_plane_range = 1  # 平面两侧3mm范围（单位：米）
#     downsample_voxel = 1  # 降采样体素大小
#
#     # 1. 读取原始点云
#     original_pcd = read_point_cloud(input_file)
#
#     # 2. 降采样加速平面检测
#     downpcd = downsample_point_cloud(original_pcd, voxel_size=downsample_voxel)
#
#     # 3. 检测最大平面
#     plane_model = find_largest_plane(downpcd)
#
#     # 4. 确定远侧方向
#     far_side_sign = determine_far_side_from_origin(plane_model)
#     print(f"平面距离原点较远的一侧: {'正方向' if far_side_sign > 0 else '负方向'}")
#
#     # 5. 过滤点云
#     filtered_pcd = filter_points(
#         original_pcd,
#         plane_model,
#         far_side_sign,
#         delete_plane_range=delete_plane_range
#     )
#
#     # 6. 保存和可视化
#     save_filtered_point_cloud(filtered_pcd, output_file)
#     visualize_result(filtered_pcd)
#
#     print(f"\n总处理时间: {time.time() - total_start:.2f}s")
#
#
# if __name__ == "__main__":
#     main()
import open3d as o3d
import numpy as np
import os
import time
import glob


# ====================== 基础功能 ======================
def read_point_cloud(file_path):
    """读取点云文件"""
    start_time = time.time()
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    data = np.loadtxt(file_path, dtype=np.float32)
    if data.size == 0 or data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"无效格式，需至少3列数据，实际{data.shape[1]}列")

    points = data[:, :3]
    print(f"[{os.path.basename(file_path)}] 原始点云: {len(points)} 个点（读取耗时: {time.time() - start_time:.2f}s）")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def downsample_point_cloud(pcd, voxel_size=0.01):
    """降采样加速平面检测"""
    start_time = time.time()
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"  降采样后: {len(downpcd.points)} 个点（耗时: {time.time() - start_time:.2f}s）")
    return downpcd


def find_largest_plane(pcd, distance_threshold=0.0015, ransac_n=5, num_iterations=500):
    """检测最大平面"""
    start_time = time.time()
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    a, b, c, d = plane_model
    print(f"  平面检测完成（耗时: {time.time() - start_time:.2f}s）")
    print(f"  平面方程: {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")
    return plane_model


def determine_far_side_from_origin(plane_model):
    """确定平面距离原点较远的一侧（返回符号）"""
    a, b, c, d = plane_model
    # d 越大（负号在方程中）表示平面离原点越远，取相反符号即可
    return -np.sign(d)


def filter_points(original_pcd, plane_model, far_side_sign, delete_plane_range=0.003):
    """
    过滤点云：
    1. 删除平面两侧 ±delete_plane_range（默认 3 mm）范围内的点
    2. 删除平面距离原点较远一侧的所有点
    保留：距离原点较近一侧且距离平面 > delete_plane_range 的点
    """
    start_time = time.time()
    a, b, c, d = plane_model
    points = np.asarray(original_pcd.points)

    signed_dist = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
    abs_dist = np.abs(signed_dist)

    keep_mask = (np.sign(signed_dist) != far_side_sign) & (abs_dist > delete_plane_range)
    filtered_points = points[keep_mask]

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    print(f"  点云过滤完成（耗时: {time.time() - start_time:.2f}s）")
    print(f"  删除平面±{delete_plane_range:.1f} mm范围及远侧点，保留 {len(filtered_pcd.points)} 个点")
    return filtered_pcd


def visualize_result(filtered_pcd, vis_voxel_size=0.02):
    """可视化处理结果（仅在需要时调用）"""
    vis_pcd = filtered_pcd.voxel_down_sample(voxel_size=vis_voxel_size)
    vis_pcd.paint_uniform_color([0, 0.6, 1])

    o3d.visualization.draw_geometries(
        [vis_pcd],
        window_name="点云处理结果（保留近侧且远离平面的点）",
        width=1280,
        height=960,
    )


def save_filtered_point_cloud(pcd, output_path):
    """保存过滤后的点云"""
    start_time = time.time()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savetxt(output_path, np.asarray(pcd.points), fmt="%.6f")
    print(f"  结果保存至 {output_path}（耗时: {time.time() - start_time:.2f}s）")


# ====================== 批量处理入口 ======================
def process_one_file(input_path, output_path,
                     downsample_voxel=1.0,
                     delete_plane_range=0.003,
                     visualize=False):
    """
    对单个 txt 点云文件执行完整流程。
    参数说明：
        downsample_voxel      – 降采样体素（单位：米），建议 0.5\~2.0，数值越大越快但精度下降
        delete_plane_range   – 平面两侧删除阈值（米），默认 3 mm
        visualize            – 是否弹出可视化窗口
    """
    print(f"\n=== 处理文件: {input_path} ===")
    # 1. 读取原始点云
    original_pcd = read_point_cloud(input_path)

    # 2. 降采样加速平面检测
    downpcd = downsample_point_cloud(original_pcd, voxel_size=downsample_voxel)

    # 3. 检测最大平面
    plane_model = find_largest_plane(downpcd)

    # 4. 确定远侧方向
    far_side_sign = determine_far_side_from_origin(plane_model)
    print(f"  平面距离原点较远的一侧: {'正方向' if far_side_sign > 0 else '负方向'}")

    # 5. 过滤点云
    filtered_pcd = filter_points(
        original_pcd,
        plane_model,
        far_side_sign,
        delete_plane_range=delete_plane_range,
    )

    # 6. 保存结果
    save_filtered_point_cloud(filtered_pcd, output_path)

    # 7. 可视化（可选）
    if visualize:
        visualize_result(filtered_pcd)


def batch_process(input_root, output_root,
                  downsample_voxel=1.0,
                  delete_plane_range=3,
                  visualize_last=True):
    """
    遍历 input_root 下所有 *.txt 文件，保持目录结构写入 output_root。
    参数：
        downsample_voxel   – 降采样体素大小（米）
        delete_plane_range – 平面两侧删除阈值（米）
        visualize_last     – 只对最后一个文件弹出可视化窗口，设 False 则全部不弹窗
    """
    txt_files = glob.glob(os.path.join(input_root, "**/*.txt"), recursive=True)
    if not txt_files:
        print("未在指定目录找到任何 txt 点云文件。")
        return

    total_start = time.time()
    for idx, in_path in enumerate(txt_files, 1):
        # 计算对应的输出路径，保持相对结构
        rel_path = os.path.relpath(in_path, input_root)
        out_path = os.path.join(output_root, rel_path)

        # 只在最后一个文件时可视化（如果需要全部可视化，把 visualize_last 改为 False 并自行调用 visualize）
        visualize = visualize_last and (idx == len(txt_files))

        process_one_file(
            input_path=in_path,
            output_path=out_path,
            downsample_voxel=downsample_voxel,
            delete_plane_range=delete_plane_range,
            visualize=visualize,
        )
    print(f"\n全部完成！共处理 {len(txt_files)} 个文件，耗时 {time.time() - total_start:.2f}s")


# ====================== 主函数 ======================
if __name__ == "__main__":
    # ------------------- 参数区 -------------------
    # 1）原始点云所在根目录（递归搜索 *.txt）
    input_root = "1107\\txt"          # ← 请改成你的输入根目录

    # 2）处理后点云保存的根目录
    output_root = "1107\\out"          # ← 请改成你想要的输出根目录

    # 3）核心算法参数（可根据实际点云密度微调）
    downsample_voxel = 1.0      # 降采样体素（米），数值越大越快
    delete_plane_range = 80 # 平面两侧删除阈值（米），默认

    # 4）是否只对最后一个文件弹出可视化窗口
    visualize_last = True

    # ------------------- 执行 -------------------
    batch_process(
        input_root,
        output_root,
        downsample_voxel=downsample_voxel,
        delete_plane_range=delete_plane_range,
        visualize_last=visualize_last,
    )
