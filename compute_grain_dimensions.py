#!/usr/bin/env python3
"""
穗粒尺寸测量脚本

读取 fused_colored_pointcloud_final.txt（格式: X Y Z Ins Sem），
每个 Ins 对应一个穗粒。用 fit_percentile_ellipsoid（PCA + 2.5%–97.5% 分位）
拟合罩住大部分点的椭球，再取轴向包络边长为粒长/粒宽/粒高（不显式剔除离散点）。
视频展示与测量共用同一套椭球逻辑，再以外接方框框住该椭球。
宽高比越接近 1 越圆润。
对全体穗粒的 |宽高比-1| 做 Z-score 归一化；
结实条件：Z-score <= 阈值视为结实；
若 Z-score > 阈值，则最短边大于全体最短边均值时仍视为结实，否则不结实。
导出 Excel 及带 Survival_rate 的 TXT 文件（TXT 保留全部原始点）。
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


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


def fit_percentile_ellipsoid(
    points: np.ndarray,
    low_percentile: float = 2.5,
    high_percentile: float = 97.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    椭球拟合（测量与视频共用）：
    1) PCA 定三主轴；
    2) 各轴投影取 [low, high] 百分位，得到罩住大部分点的轴向包络；
    3) 该包络即为椭球的轴向半轴范围，外接方框 = 该 mins/maxs。

    返回 (质心, 主轴3x3, 局部mins, 局部maxs)；点数不足或退化时返回 None。
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


def compute_ellipsoid_dimensions(
    points: np.ndarray,
    low_percentile: float = 2.5,
    high_percentile: float = 97.5,
) -> tuple[float, float, float]:
    """
    对点云做三维椭球拟合，返回 (粒长, 粒宽, 粒高)，按从大到小排序。
    与视频展示共用 fit_percentile_ellipsoid：椭球罩大部分点，尺寸=轴向包络边长。
    """
    if len(points) == 0:
        return 0.0, 0.0, 0.0

    fitted = fit_percentile_ellipsoid(
        points,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
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
    """宽高比偏离度 Z-score <= 阈值时通过形状条件。"""
    return normalized_deviation <= z_threshold


def is_grain_firm_by_shortest(
    shortest: float,
    mean_shortest: float,
) -> bool:
    """最短边 > 全体最短边均值时通过最短边条件。"""
    if mean_shortest <= 0:
        return shortest > 0
    return shortest > mean_shortest


def is_grain_firm(
    normalized_deviation: float,
    shortest: float,
    mean_shortest: float,
    deviation_z_threshold: float = 1.5,
) -> tuple[bool, str]:
    """
    结实判定：
    - Z-score <= 阈值 → 结实
    - Z-score > 阈值 且 最短边 > 全体最短边均值 → 结实
    - Z-score > 阈值 且 最短边 <= 全体最短边均值 → 不结实
    """
    if normalized_deviation <= deviation_z_threshold:
        return True, "结实"

    # Z-score 偏大：用最短边做二次判定
    if mean_shortest <= 0:
        firm_by_shortest = shortest > 0
    else:
        firm_by_shortest = shortest > mean_shortest

    if firm_by_shortest:
        return True, "结实(最短边偏大)"
    return False, "宽高比偏离且最短边偏短"


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
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, int], np.ndarray, np.ndarray, np.ndarray]:
    """
    测量所有穗粒（椭球拟合 + 百分位轴向包络），再判定结实。
    结实条件：Z-score<=阈值；或 Z-score>阈值 但最短边>全体最短边均值。
    """
    unique_ids = np.sort(np.unique(instance_ids))
    grain_records: list[dict] = []
    total_removed = 0
    global_keep_mask = np.zeros(len(points), dtype=bool)

    print(
        f"离散点过滤已关闭：椭球拟合用全部点，"
        f"轴向尺寸取 {extent_low_percentile:g}%–{extent_high_percentile:g}% 分位跨度..."
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

        length, width, height = compute_ellipsoid_dimensions(
            filtered_points,
            low_percentile=extent_low_percentile,
            high_percentile=extent_high_percentile,
        )
        shortest = min(length, width, height)
        wh_ratio = compute_width_height_ratio(width, height)
        deviation = compute_deviation_from_one(wh_ratio)

        grain_records.append({
            "序号": idx,
            "实例ID": int(ins_id),
            "原始点数": original_count,
            "有效点数": len(filtered_points),
            "剔除点数": removed_count,
            "粒长": round(length, 4),
            "粒宽": round(width, 4),
            "粒高": round(height, 4),
            "最短边": round(shortest, 4),
            "长宽比": round(length / width, 3) if width > 0 else 0.0,
            "长高比": round(length / height, 3) if height > 0 else 0.0,
            "宽高比": round(wh_ratio, 3),
            "偏离度": round(deviation, 4),
        })

        if idx % 10 == 0 or idx == len(unique_ids):
            print(f"  已处理 {idx}/{len(unique_ids)} 个实例")

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
            record["最短边"],
            mean_shortest,
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
            f"Z-score<={deviation_z_threshold} 视为结实；"
            f"Z-score>{deviation_z_threshold} 时若最短边>{round(mean_shortest, 4)} 仍结实，否则不结实"
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
    """导出过滤后的点云 Survival TXT（结实=1，不结实=0）。"""
    txt_path = Path(txt_path)
    temp_path = txt_path.with_suffix(txt_path.suffix + ".tmp")

    with open(temp_path, "w", encoding="utf-8") as f:
        f.write("// X Y Z Ins Sem Survival_rate\n")
        f.write(f"// Overall_Survival_rate: {overall_survival_rate:.4f}\n")
        f.write(f"// Filtered_point_count: {len(points)}\n")
        for i in range(len(points)):
            ins_id = int(instance_ids[i])
            survival = survival_map.get(ins_id, 0)
            f.write(
                f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                f"{ins_id} {int(sem[i])} {survival}\n"
            )

    atomic_replace(temp_path, txt_path, "文件")
    print(f"Survival TXT 已保存: {txt_path}（过滤后 {len(points)} 个点）")


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
        r"E:\rice\ply\013\fused_colored_pointcloud_final.txt"
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
        help="宽高比偏离度的 Z-score 阈值，超过则判定为非结实（默认 1.5）",
    )
    parser.add_argument(
        "--extent_low_percentile",
        type=float,
        default=2.5,
        help="椭球轴向尺寸下百分位（默认 2.5，与高百分位组成分位包络）",
    )
    parser.add_argument(
        "--extent_high_percentile",
        type=float,
        default=97.5,
        help="椭球轴向尺寸上百分位（默认 97.5）",
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
    )

    summary = dict(zip(summary_df["指标"], summary_df["数值"]))
    overall_survival_rate = summary["结实率(%)"] / 100.0

    save_excel(detail_df, summary_df, str(output_excel))
    save_survival_txt(
        str(output_txt),
        filtered_points,
        filtered_instance_ids,
        filtered_sem,
        survival_map,
        overall_survival_rate,
    )
    print_summary(summary_df)


if __name__ == "__main__":
    main()
