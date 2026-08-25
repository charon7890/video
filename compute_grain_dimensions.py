#!/usr/bin/env python3
"""
穗粒尺寸测量脚本

读取 fused_colored_pointcloud_final.txt（格式: X Y Z Ins Sem），
每个 Ins 对应一个穗粒。默认用快速 PCA + 分位包络拟合椭球并量外接方框；
先据此判定结实，仅对不结实的粒改用最小体积椭球（MVEE）重测后再定最终结果。
视频展示与测量共用同一套椭球接口。宽高比越接近 1 越圆润。
对全体穗粒的 |宽高比-1| 做 Z-score 归一化；
结实条件：宽高比偏离度 Z-score <= 阈值视为结实，否则不结实。
导出 Excel 及带 Survival_rate 的 TXT（含 RGB：结实绿 / 非结实红，TXT 保留全部原始点）。
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


def load_pointcloud_txt(txt_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载点云文本文件，返回坐标 (N,3)、实例 ID (N,) 和 Sem (N,)。"""
    print(f"正在加载点云: {txt_path}")
    data = np.loadtxt(txt_path, skiprows=1)
    points = data[:, :3]
    instance_ids = data[:, 3].astype(int)
    sem = data[:, 4].astype(int) if data.shape[1] > 4 else np.zeros(len(points), dtype=int)
    print(f"已加载 {len(points)} 个点，{len(np.unique(instance_ids))} 个实例")
    return points, instance_ids, sem


def _knn_mean_distances(points: np.ndarray, k: int) -> np.ndarray:
    """计算每个点到其 k 个最近邻的平均距离。"""
    n = len(points)
    if n <= 1:
        return np.zeros(n, dtype=np.float64)

    k_eff = max(1, min(k, n - 1))
    # 点数较少时直接用全距离矩阵；较大时分块，避免内存暴涨
    if n <= 4000:
        diff = points[:, None, :] - points[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        knn = np.partition(dist, k_eff, axis=1)[:, :k_eff]
        return knn.mean(axis=1)

    mean_dists = np.empty(n, dtype=np.float64)
    chunk = 512
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        diff = points[start:end, None, :] - points[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        for i, global_i in enumerate(range(start, end)):
            dist[i, global_i] = np.inf
        knn = np.partition(dist, k_eff, axis=1)[:, :k_eff]
        mean_dists[start:end] = knn.mean(axis=1)
    return mean_dists


def filter_statistical_outliers(
    points: np.ndarray,
    knn_k: int = 20,
    std_ratio: float = 2.0,
) -> np.ndarray:
    """
    统计离散点过滤（类似 Point Cloud Statistical Outlier Removal）：
    若某点到 k 近邻的平均距离 > 全体均值 + std_ratio * 标准差，则剔除。
    返回布尔掩码（True=保留）。
    """
    n = len(points)
    if n < 3:
        return np.ones(n, dtype=bool)

    mean_dists = _knn_mean_distances(points, knn_k)
    global_mean = float(mean_dists.mean())
    global_std = float(mean_dists.std())
    threshold = global_mean + std_ratio * global_std
    inlier_mask = mean_dists <= threshold

    if not np.any(inlier_mask):
        keep_count = max(1, n // 2)
        nearest_indices = np.argsort(mean_dists)[:keep_count]
        inlier_mask = np.zeros(n, dtype=bool)
        inlier_mask[nearest_indices] = True

    return inlier_mask


def filter_center_outliers(
    points: np.ndarray,
    distance_percentile: float = 95.0,
    std_ratio: float = 2.0,
) -> np.ndarray:
    """剔除偏离质心过远的散点，返回布尔掩码（True=保留）。"""
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=bool)

    centroid = points.mean(axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)

    percentile_threshold = np.percentile(distances, distance_percentile)
    std_threshold = distances.mean() + std_ratio * distances.std()
    threshold = min(percentile_threshold, std_threshold)

    inlier_mask = distances <= threshold
    if not np.any(inlier_mask):
        keep_count = max(1, n // 2)
        nearest_indices = np.argsort(distances)[:keep_count]
        inlier_mask = np.zeros(n, dtype=bool)
        inlier_mask[nearest_indices] = True

    return inlier_mask


def filter_discrete_points(
    points: np.ndarray,
    knn_k: int = 20,
    statistical_std_ratio: float = 2.0,
    distance_percentile: float = 95.0,
    center_std_ratio: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    离散点过滤（已关闭）：椭球拟合直接使用全部点，不剔除。
    返回 (点云, 全 True 掩码, 原始点数, 剔除点数=0)。
    """
    del knn_k, statistical_std_ratio, distance_percentile, center_std_ratio
    original_count = len(points)
    if original_count == 0:
        return points, np.zeros(0, dtype=bool), 0, 0
    keep_mask = np.ones(original_count, dtype=bool)
    return points, keep_mask, original_count, 0


def compute_bbox_dimensions(points: np.ndarray) -> tuple[float, float, float]:
    """计算 AABB 尺寸，返回 (粒长, 粒宽, 粒高)，按从大到小排序。"""
    if len(points) == 0:
        return 0.0, 0.0, 0.0

    extents = points.max(axis=0) - points.min(axis=0)
    sorted_extents = np.sort(extents)[::-1]
    return float(sorted_extents[0]), float(sorted_extents[1]), float(sorted_extents[2])


def _select_points_for_min_volume_ellipsoid(
    points: np.ndarray,
    coverage: float = 0.95,
) -> np.ndarray:
    """
    快速保留 coverage 比例的主体点：最多两轮马氏距离筛选（不再逐步剥点）。
    """
    n = len(points)
    if n <= 4:
        return points

    keep_n = max(4, int(np.ceil(n * float(np.clip(coverage, 0.5, 1.0)))))
    keep_n = min(keep_n, n)
    pts = np.asarray(points, dtype=np.float64)

    for _ in range(2):
        if len(pts) <= keep_n:
            break
        center = pts.mean(axis=0)
        centered = pts - center
        cov = np.cov(centered.T) + np.eye(3) * 1e-12
        if not np.isfinite(cov).all():
            break
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            break
        mahal = np.einsum("ij,jk,ik->i", centered, inv_cov, centered)
        keep_idx = np.argpartition(mahal, keep_n - 1)[:keep_n]
        pts = pts[keep_idx]
    return pts


def _khachiyan_mvee(
    points: np.ndarray,
    tol: float = 1e-3,
    max_iter: int = 400,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Khachiyan 算法：求包含给定点集的最小体积椭球。
    返回 (中心 c, 矩阵 A)，椭球定义 (x-c)^T A (x-c) <= 1。
    """
    pts = np.asarray(points, dtype=np.float64)
    n, d = pts.shape
    if n < d + 1:
        return None

    q = np.empty((d + 1, n), dtype=np.float64)
    q[:d, :] = pts.T
    q[d, :] = 1.0
    u = np.full(n, 1.0 / n, dtype=np.float64)
    eye = np.eye(d + 1)

    for _ in range(max_iter):
        x_mat = (q * u) @ q.T + eye * 1e-12
        try:
            inv_x = np.linalg.inv(x_mat)
        except np.linalg.LinAlgError:
            return None
        # M_i = q_i^T inv(X) q_i
        y = inv_x @ q
        m = np.sum(q * y, axis=0)
        j = int(np.argmax(m))
        max_m = float(m[j])
        if max_m <= d + 1.0 + tol:
            break
        step = (max_m - d - 1.0) / ((d + 1.0) * (max_m - 1.0))
        new_u = (1.0 - step) * u
        new_u[j] += step
        if float(np.linalg.norm(new_u - u)) <= tol:
            u = new_u
            break
        u = new_u

    center = pts.T @ u
    p_u = pts * u[:, None]
    try:
        a_mat = np.linalg.inv(pts.T @ p_u - np.outer(center, center)) / d
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(a_mat).all():
        return None
    return center.astype(np.float64), a_mat.astype(np.float64)


def _subsample_for_hull(points: np.ndarray, max_points: int = 1500) -> np.ndarray:
    """凸包前对超大点集做均匀下采样，显著加速。"""
    n = len(points)
    if n <= max_points:
        return points
    idx = np.linspace(0, n - 1, max_points, dtype=np.int64)
    return points[idx]


def fit_min_volume_ellipsoid(
    points: np.ndarray,
    coverage: float = 0.95,
    tol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    最小体积椭球拟合（测量与视频共用）：
    1) 保留 coverage 比例的主体点（去掉最远飞点）；
    2) 对凸包顶点求 MVEE（最小体积包含椭球）；
    3) 外接方框 = 椭球主轴方向上的包围盒（边长=2×半轴）。

    返回 (中心, 主轴3x3, 局部mins, 局部maxs)；失败返回 None。
    """
    if len(points) < 3:
        return None

    core = _select_points_for_min_volume_ellipsoid(points, coverage=coverage)
    if len(core) < 4:
        center = core.mean(axis=0)
        centered = core - center
        cov = np.cov(centered.T) if len(core) >= 3 else np.eye(3)
        if not np.isfinite(cov).all() or abs(float(np.linalg.det(cov))) < 1e-18:
            return None
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        axes = evecs[:, order]
        if np.linalg.det(axes) < 0:
            axes[:, -1] *= -1
        local = centered @ axes
        mins = local.min(axis=0)
        maxs = local.max(axis=0)
        return center.astype(np.float64), axes.astype(np.float64), mins, maxs

    hull_pts = _subsample_for_hull(core, max_points=1500)
    try:
        hull = ConvexHull(hull_pts)
        verts = hull_pts[hull.vertices]
    except Exception:
        verts = hull_pts

    if len(verts) < 4:
        verts = hull_pts

    # 凸包顶点过多时再抽样，控制 Khachiyan 规模
    if len(verts) > 128:
        v_idx = np.linspace(0, len(verts) - 1, 128, dtype=np.int64)
        verts = verts[v_idx]

    mvee = _khachiyan_mvee(verts, tol=tol, max_iter=400)
    if mvee is None:
        return None
    center, a_mat = mvee

    try:
        shape = np.linalg.inv(a_mat)
    except np.linalg.LinAlgError:
        return None
    evals, evecs = np.linalg.eigh(shape)
    evals = np.maximum(evals, 1e-18)
    order = np.argsort(evals)[::-1]
    axes = evecs[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    semi_axes = np.sqrt(evals[order])

    mins = -semi_axes
    maxs = semi_axes
    return (
        center.astype(np.float64),
        axes.astype(np.float64),
        mins.astype(np.float64),
        maxs.astype(np.float64),
    )


def fit_pca_percentile_ellipsoid(
    points: np.ndarray,
    low_percentile: float = 2.5,
    high_percentile: float = 97.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    快速椭球拟合：PCA 主轴 + 轴向百分位包络（默认路径）。
    返回 (质心, 主轴3x3, 局部mins, 局部maxs)。
    """
    if len(points) < 3:
        return None

    center = points.mean(axis=0)
    centered = points - center
    cov = np.cov(centered.T)
    if not np.isfinite(cov).all() or abs(float(np.linalg.det(cov))) < 1e-18:
        return None

    _eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(_eigenvalues)[::-1]
    axes = eigenvectors[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1

    local = centered @ axes
    if len(points) < 20:
        mins = local.min(axis=0)
        maxs = local.max(axis=0)
    else:
        low = float(np.clip(low_percentile, 0.0, 49.0))
        high = float(np.clip(high_percentile, 51.0, 100.0))
        if high <= low:
            low, high = 2.5, 97.5
        mins = np.percentile(local, low, axis=0)
        maxs = np.percentile(local, high, axis=0)

    return (
        center.astype(np.float64),
        axes.astype(np.float64),
        np.asarray(mins, dtype=np.float64),
        np.asarray(maxs, dtype=np.float64),
    )


def fit_percentile_ellipsoid(
    points: np.ndarray,
    low_percentile: float = 2.5,
    high_percentile: float = 97.5,
    coverage: float | None = None,
    use_mvee: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    默认：快速 PCA + 分位包络。
    use_mvee=True 时：最小体积椭球（仅用于粒长异常偏长的实例）。
    """
    if use_mvee:
        if coverage is None:
            coverage = float(np.clip((high_percentile - low_percentile) / 100.0, 0.5, 1.0))
        return fit_min_volume_ellipsoid(points, coverage=coverage)
    return fit_pca_percentile_ellipsoid(
        points,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
    )


def compute_ellipsoid_dimensions(
    points: np.ndarray,
    low_percentile: float = 2.5,
    high_percentile: float = 97.5,
    coverage: float | None = None,
    use_mvee: bool = False,
) -> tuple[float, float, float]:
    """
    椭球外接方框三边 → (粒长, 粒宽, 粒高)。
    默认快速 PCA；use_mvee=True 时用最小体积椭球。
    """
    if len(points) == 0:
        return 0.0, 0.0, 0.0

    fitted = fit_percentile_ellipsoid(
        points,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        coverage=coverage,
        use_mvee=use_mvee,
    )
    if fitted is None:
        return compute_bbox_dimensions(points)

    _center, _axes, mins, maxs = fitted
    extents = maxs - mins
    sorted_extents = np.sort(extents)[::-1]
    return float(sorted_extents[0]), float(sorted_extents[1]), float(sorted_extents[2])


def compute_width_height_ratio(width: float, height: float) -> float:
    """计算宽高比（粒宽/粒高），越接近 1 表示截面越圆润。"""
    if height <= 0:
        return 0.0
    return float(width / height)


def compute_deviation_from_one(width_height_ratio: float) -> float:
    """计算宽高比偏离 1 的程度。"""
    return float(abs(width_height_ratio - 1.0))


def normalize_deviations_zscore(deviations: list[float]) -> tuple[list[float], float, float]:
    """对偏离度做 Z-score 归一化，返回 (z_scores, mean, std)。"""
    if not deviations:
        return [], 0.0, 0.0
    arr = np.asarray(deviations, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std())
    if std <= 0:
        return [0.0] * len(deviations), mean, std
    z_scores = ((arr - mean) / std).tolist()
    return z_scores, mean, std


def is_grain_firm_by_deviation(normalized_deviation: float, z_threshold: float = 1.5) -> bool:
    """宽高比偏离度 Z-score <= 阈值时为结实。"""
    return normalized_deviation <= z_threshold


def is_grain_firm(
    normalized_deviation: float,
    deviation_z_threshold: float = 1.5,
) -> tuple[bool, str]:
    """结实判定：仅按宽高比偏离度 Z-score。"""
    if is_grain_firm_by_deviation(normalized_deviation, deviation_z_threshold):
        return True, "结实"
    return False, "宽高比偏离过大"


def _apply_dimensions_to_record(
    rec: dict,
    length: float,
    width: float,
    height: float,
    fit_method: str,
) -> None:
    shortest = min(length, width, height)
    wh_ratio = compute_width_height_ratio(width, height)
    deviation = compute_deviation_from_one(wh_ratio)
    rec["粒长"] = round(length, 4)
    rec["粒宽"] = round(width, 4)
    rec["粒高"] = round(height, 4)
    rec["最短边"] = round(shortest, 4)
    rec["长宽比"] = round(length / width, 3) if width > 0 else 0.0
    rec["长高比"] = round(length / height, 3) if height > 0 else 0.0
    rec["宽高比"] = round(wh_ratio, 3)
    rec["偏离度"] = round(deviation, 4)
    rec["拟合方式"] = fit_method


def _preliminary_firm_flags(
    grain_records: list[dict],
    deviation_z_threshold: float,
) -> list[bool]:
    """
    初判结实：只按宽高比偏离度 Z-score，不使用最短边二次条件。
    Z-score <= 阈值 → 结实；否则 → 不结实（进入 MVEE 重测）。
    """
    deviations = [g["偏离度"] for g in grain_records]
    z_scores, _mean_dev, _std_dev = normalize_deviations_zscore(deviations)
    return [
        is_grain_firm_by_deviation(z_score, deviation_z_threshold)
        for z_score in z_scores
    ]


def measure_all_grains(
    points: np.ndarray,
    instance_ids: np.ndarray,
    sem: np.ndarray,
    distance_percentile: float = 95.0,
    std_ratio: float = 2.0,
    knn_k: int = 20,
    statistical_std_ratio: float = 2.0,
    deviation_z_threshold: float = 1.5,
    extent_low_percentile: float = 2.5,
    extent_high_percentile: float = 97.5,
    ellipsoid_coverage: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, int], np.ndarray, np.ndarray, np.ndarray]:
    """
    测量所有穗粒：默认快速 PCA；初判为不结实的再用 MVEE 重测，然后做最终结实判定。
    """
    unique_ids = np.sort(np.unique(instance_ids))
    grain_records: list[dict] = []
    grain_points_list: list[np.ndarray] = []
    total_removed = 0
    global_keep_mask = np.zeros(len(points), dtype=bool)

    coverage = float(ellipsoid_coverage)
    if coverage <= 0:
        coverage = float(np.clip((extent_high_percentile - extent_low_percentile) / 100.0, 0.5, 1.0))

    print(
        f"椭球拟合：默认 PCA+{extent_low_percentile:g}%–{extent_high_percentile:g}% 分位；"
        f"仅不结实粒改用最小体积椭球重测(coverage={coverage:.0%})..."
    )
    for idx, ins_id in enumerate(unique_ids, start=1):
        inst_mask = instance_ids == ins_id
        inst_indices = np.flatnonzero(inst_mask)
        inst_points = points[inst_indices]
        filtered_points, local_keep_mask, original_count, removed_count = filter_discrete_points(
            inst_points,
            knn_k=knn_k,
            statistical_std_ratio=statistical_std_ratio,
            distance_percentile=distance_percentile,
            center_std_ratio=std_ratio,
        )
        global_keep_mask[inst_indices[local_keep_mask]] = True
        total_removed += removed_count
        grain_points_list.append(filtered_points)

        length, width, height = compute_ellipsoid_dimensions(
            filtered_points,
            low_percentile=extent_low_percentile,
            high_percentile=extent_high_percentile,
            use_mvee=False,
        )
        grain_records.append({
            "序号": idx,
            "实例ID": int(ins_id),
            "原始点数": original_count,
            "有效点数": len(filtered_points),
            "剔除点数": removed_count,
            "粒长": 0.0,
            "粒宽": 0.0,
            "粒高": 0.0,
            "最短边": 0.0,
            "长宽比": 0.0,
            "长高比": 0.0,
            "宽高比": 0.0,
            "偏离度": 0.0,
            "拟合方式": "PCA分位",
        })
        _apply_dimensions_to_record(grain_records[-1], length, width, height, "PCA分位")

        if idx % 10 == 0 or idx == len(unique_ids):
            print(f"  已处理 {idx}/{len(unique_ids)} 个实例")

    firm_flags = _preliminary_firm_flags(grain_records, deviation_z_threshold)
    hollow_indices = [i for i, firm in enumerate(firm_flags) if not firm]
    if hollow_indices:
        print(f"初判不结实 {len(hollow_indices)} 颗，改用最小体积椭球重测...")
        for i in hollow_indices:
            old_len = grain_records[i]["粒长"]
            length, width, height = compute_ellipsoid_dimensions(
                grain_points_list[i],
                coverage=coverage,
                use_mvee=True,
            )
            _apply_dimensions_to_record(grain_records[i], length, width, height, "MVEE")
            print(
                f"  重测不结实实例 #{grain_records[i]['实例ID']}: "
                f"粒长 {old_len:.4f} → {length:.4f}"
            )
    else:
        print("初判无无结实粒，全部使用快速 PCA 拟合")

    print(f"测量完成，保留全部 {int(global_keep_mask.sum())} 个点（未剔除散点）")

    deviations = [g["偏离度"] for g in grain_records]
    shortest_list = [g["最短边"] for g in grain_records]
    z_scores, mean_dev, std_dev = normalize_deviations_zscore(deviations)
    mean_shortest = float(np.mean(shortest_list)) if shortest_list else 0.0

    rows = []
    survival_map: dict[int, int] = {}
    for record, z_score in zip(grain_records, z_scores):
        is_firm, reason = is_grain_firm(
            z_score,
            deviation_z_threshold,
        )
        survival = 1 if is_firm else 0
        survival_map[record["实例ID"]] = survival

        row = {k: v for k, v in record.items() if k != "实例ID"}
        row["归一化偏离度"] = round(z_score, 3)
        row["圆润度"] = round(1.0 / (1.0 + record["偏离度"]), 3)
        row["最短边/均值"] = (
            round(record["最短边"] / mean_shortest, 3) if mean_shortest > 0 else 0.0
        )
        row["是否结实"] = "是" if is_firm else "否"
        row["判定说明"] = reason
        rows.append(row)

    detail_df = pd.DataFrame(rows)
    summary_df = build_summary_df(
        detail_df,
        mean_dev,
        std_dev,
        deviation_z_threshold,
        mean_shortest,
        total_removed,
    )
    filtered_points = points[global_keep_mask]
    filtered_instance_ids = instance_ids[global_keep_mask]
    filtered_sem = sem[global_keep_mask]
    return detail_df, summary_df, survival_map, filtered_points, filtered_instance_ids, filtered_sem


def build_summary_df(
    detail_df: pd.DataFrame,
    mean_deviation: float,
    std_deviation: float,
    deviation_z_threshold: float,
    mean_shortest: float,
    total_removed_points: int = 0,
) -> pd.DataFrame:
    """构建汇总表：总粒数、结实粒数、结实率。"""
    total = len(detail_df)
    firm_count = int((detail_df["是否结实"] == "是").sum())
    firm_rate = round(firm_count / total * 100, 2) if total else 0.0

    return pd.DataFrame([
        {"指标": "总粒数", "数值": total},
        {"指标": "结实粒数", "数值": firm_count},
        {"指标": "结实率(%)", "数值": firm_rate},
        {"指标": "离散点剔除总数", "数值": total_removed_points},
        {"指标": "偏离度均值", "数值": round(mean_deviation, 4)},
        {"指标": "偏离度标准差", "数值": round(std_deviation, 4)},
        {"指标": "归一化偏离阈值", "数值": deviation_z_threshold},
        {"指标": "最短边均值", "数值": round(mean_shortest, 4)},
        {"指标": "结实判定条件", "数值": (
            f"仅宽高比偏离度 Z-score<={deviation_z_threshold} 为结实"
        )},
    ])


def resolve_output_paths(
    input_path: Path,
    output_excel: str | None,
    output_txt: str | None,
) -> tuple[Path, Path]:
    """根据输入文件名生成默认输出路径。"""
    stem = input_path.stem
    parent = input_path.parent
    excel_path = Path(output_excel) if output_excel else parent / f"{stem}.xlsx"
    txt_path = Path(output_txt) if output_txt else parent / f"{stem}_survival.txt"
    return excel_path, txt_path


def atomic_replace(temp_path: Path, final_path: Path, file_kind: str) -> None:
    """先写临时文件，再覆盖目标文件。"""
    try:
        if final_path.exists():
            final_path.unlink()
        os.replace(temp_path, final_path)
    except PermissionError as exc:
        temp_path.unlink(missing_ok=True)
        raise PermissionError(
            f"无法覆盖 {final_path}，请先在 Excel 中关闭该{file_kind}后再运行。"
        ) from exc


def save_survival_txt(
    txt_path: str,
    points: np.ndarray,
    instance_ids: np.ndarray,
    sem: np.ndarray,
    survival_map: dict[int, int],
    overall_survival_rate: float,
) -> None:
    """
    导出 Survival TXT：X Y Z R G B Ins Sem Survival_rate
    结实 → 绿色，非结实 → 红色（便于 CloudCompare 直接按 RGB 显示）。
    """
    txt_path = Path(txt_path)
    temp_path = txt_path.with_suffix(txt_path.suffix + ".tmp")

    n = len(points)
    survival = np.fromiter(
        (int(survival_map.get(int(ins_id), 0)) for ins_id in instance_ids),
        dtype=np.int32,
        count=n,
    )
    rgb = np.empty((n, 3), dtype=np.int32)
    firm_mask = survival == 1
    rgb[firm_mask] = (40, 200, 80)      # 结实：绿
    rgb[~firm_mask] = (220, 50, 50)     # 非结实：红

    out = np.column_stack([
        points.astype(np.float64),
        rgb,
        instance_ids.astype(np.int32),
        sem.astype(np.int32),
        survival,
    ])

    header = (
        "X Y Z R G B Ins Sem Survival_rate\n"
        f"Overall_Survival_rate: {overall_survival_rate:.4f}\n"
        f"Filtered_point_count: {n}\n"
        "Color: firm=green(40,200,80) hollow=red(220,50,50)"
    )
    np.savetxt(
        temp_path,
        out,
        fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d", "%d", "%d", "%d"],
        header=header,
        comments="// ",
    )

    atomic_replace(temp_path, txt_path, "文件")
    print(f"Survival TXT 已保存: {txt_path}（{n} 个点，结实绿/非结实红）")


def save_excel(detail_df: pd.DataFrame, summary_df: pd.DataFrame, excel_path: str) -> None:
    """导出 Excel，先写临时文件再覆盖，避免旧文件占用导致失败。"""
    excel_path = Path(excel_path)
    temp_path = excel_path.with_suffix(".tmp.xlsx")

    try:
        with pd.ExcelWriter(temp_path, engine="openpyxl") as writer:
            detail_df.to_excel(writer, sheet_name="穗粒明细", index=False)
            summary_df.to_excel(writer, sheet_name="汇总", index=False)
        atomic_replace(temp_path, excel_path, "Excel 文件")
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise

    print(f"Excel 已保存: {excel_path}")


def print_summary(summary_df: pd.DataFrame) -> None:
    """打印汇总信息。"""
    summary = dict(zip(summary_df["指标"], summary_df["数值"]))
    print("\n" + "=" * 48)
    print("穗粒测量完成")
    print("=" * 48)
    print(f"总粒数: {int(summary['总粒数'])}")
    print(f"结实粒数: {int(summary['结实粒数'])}")
    print(f"结实率: {summary['结实率(%)']}%")
    if "离散点剔除总数" in summary:
        print(f"离散点剔除总数: {int(summary['离散点剔除总数'])}")
    print(f"偏离度均值: {summary['偏离度均值']}")
    print(f"偏离度标准差: {summary['偏离度标准差']}")
    print(f"归一化偏离阈值: {summary['归一化偏离阈值']}")
    print(f"最短边均值: {summary['最短边均值']}")
    print(f"结实判定条件: {summary['结实判定条件']}")
    print("=" * 48)


def main():
    default_input = (
        r"E:\rice\ply\001\fused_colored_pointcloud_final.txt"  
    )

    parser = argparse.ArgumentParser(description="穗粒尺寸测量与结实率评估，导出 Excel")
    parser.add_argument("--input_txt", type=str, default=default_input, help="输入点云 TXT 文件")
    parser.add_argument("--output_excel", type=str, default=None, help="输出 Excel 路径")
    parser.add_argument("--output_txt", type=str, default=None, help="输出 Survival_rate TXT 路径")
    parser.add_argument("--distance_percentile", type=float, default=95.0, help="质心散点过滤百分位阈值")
    parser.add_argument("--std_ratio", type=float, default=2.0, help="质心散点过滤标准差倍数")
    parser.add_argument("--knn_k", type=int, default=20, help="离散点过滤的近邻数 K")
    parser.add_argument(
        "--statistical_std_ratio",
        type=float,
        default=2.0,
        help="KNN 统计离散点过滤的标准差倍数",
    )
    parser.add_argument(
        "--deviation_z_threshold",
        type=float,
        default=1.4,
        help="宽高比偏离度的 Z-score 阈值，超过则判定为非结实（默认 1.6）",
    )
    parser.add_argument(
        "--ellipsoid_coverage",
        type=float,
        default=0.95,
        help="不结实粒改用 MVEE 重测时的点覆盖比例（默认 0.95）",
    )
    parser.add_argument(
        "--extent_low_percentile",
        type=float,
        default=2.5,
        help="PCA 分位包络下百分位（默认 2.5）",
    )
    parser.add_argument(
        "--extent_high_percentile",
        type=float,
        default=97.5,
        help="PCA 分位包络上百分位（默认 97.5）",
    )
    args = parser.parse_args()

    input_path = Path(args.input_txt)
    output_excel, output_txt = resolve_output_paths(
        input_path,
        args.output_excel,
        args.output_txt,
    )

    points, instance_ids, sem = load_pointcloud_txt(str(input_path))
    (
        detail_df,
        summary_df,
        survival_map,
        filtered_points,
        filtered_instance_ids,
        filtered_sem,
    ) = measure_all_grains(
        points,
        instance_ids,
        sem,
        distance_percentile=args.distance_percentile,
        std_ratio=args.std_ratio,
        knn_k=args.knn_k,
        statistical_std_ratio=args.statistical_std_ratio,
        deviation_z_threshold=args.deviation_z_threshold,
        extent_low_percentile=args.extent_low_percentile,
        extent_high_percentile=args.extent_high_percentile,
        ellipsoid_coverage=args.ellipsoid_coverage,
    )

    summary = dict(zip(summary_df["指标"], summary_df["数值"]))
    overall_survival_rate = summary["结实率(%)"] / 100.0

    save_survival_txt(
        str(output_txt),
        filtered_points,
        filtered_instance_ids,
        filtered_sem,
        survival_map,
        overall_survival_rate,
    )
    try:
        save_excel(detail_df, summary_df, str(output_excel))
    except PermissionError as exc:
        print(f"警告: {exc}")
    print_summary(summary_df)


if __name__ == "__main__":
    main()
