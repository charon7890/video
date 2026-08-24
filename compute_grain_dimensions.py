#!/usr/bin/env python3
"""
穗粒尺寸测量脚本

读取 fused_colored_pointcloud_final.txt（格式: X Y Z Ins Sem），
每个 Ins 对应一个穗粒。通过长方体框选（AABB）计算粒长、粒宽、粒高，
框选前剔除偏离中心过远的散点；宽高比（粒宽/粒高）越接近 1 越圆润。
对全体穗粒的 |宽高比-1| 做 Z-score 归一化，明显偏离 1 的记为非结实；
粒长低于全体粒长均值指定比例的也记为非结实，
导出 Excel 及带 Survival_rate 的 TXT 文件。
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


def filter_center_outliers(
    points: np.ndarray,
    distance_percentile: float = 95.0,
    std_ratio: float = 2.0,
) -> np.ndarray:
    """剔除偏离质心过远的散点，返回有效点。"""
    if len(points) == 0:
        return points

    centroid = points.mean(axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)

    percentile_threshold = np.percentile(distances, distance_percentile)
    std_threshold = distances.mean() + std_ratio * distances.std()
    threshold = min(percentile_threshold, std_threshold)

    inlier_mask = distances <= threshold
    if not np.any(inlier_mask):
        keep_count = max(1, len(points) // 2)
        nearest_indices = np.argsort(distances)[:keep_count]
        inlier_mask = np.zeros(len(points), dtype=bool)
        inlier_mask[nearest_indices] = True

    return points[inlier_mask]


def compute_bbox_dimensions(points: np.ndarray) -> tuple[float, float, float]:
    """计算 AABB 尺寸，返回 (粒长, 粒宽, 粒高)，按从大到小排序。"""
    if len(points) == 0:
        return 0.0, 0.0, 0.0

    extents = points.max(axis=0) - points.min(axis=0)
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


def is_grain_firm_by_length(
    length: float,
    mean_length: float,
    min_length_ratio: float = 0.4,
) -> bool:
    """粒长 >= 全体粒长均值 × 比例阈值时为结实，否则为过短非结实。"""
    if mean_length <= 0:
        return length > 0
    return length >= min_length_ratio * mean_length


def is_grain_firm(
    normalized_deviation: float,
    length: float,
    mean_length: float,
    deviation_z_threshold: float = 1.5,
    min_length_ratio: float = 0.4,
) -> tuple[bool, str]:
    """综合判定穗粒是否结实，返回 (是否结实, 判定说明)。"""
    firm_by_shape = is_grain_firm_by_deviation(normalized_deviation, deviation_z_threshold)
    firm_by_length = is_grain_firm_by_length(length, mean_length, min_length_ratio)

    if firm_by_shape and firm_by_length:
        return True, "结实"
    reasons = []
    if not firm_by_shape:
        reasons.append("宽高比偏离")
    if not firm_by_length:
        reasons.append("粒长过短")
    return False, "、".join(reasons)


def measure_all_grains(
    points: np.ndarray,
    instance_ids: np.ndarray,
    distance_percentile: float = 95.0,
    std_ratio: float = 2.0,
    deviation_z_threshold: float = 1.5,
    min_length_ratio: float = 0.4,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, int]]:
    """测量所有穗粒，返回逐粒明细、汇总表、实例 Survival_rate 映射。"""
    unique_ids = np.sort(np.unique(instance_ids))
    grain_records: list[dict] = []

    for idx, ins_id in enumerate(unique_ids, start=1):
        inst_points = points[instance_ids == ins_id]
        filtered_points = filter_center_outliers(
            inst_points,
            distance_percentile=distance_percentile,
            std_ratio=std_ratio,
        )
        length, width, height = compute_bbox_dimensions(filtered_points)
        wh_ratio = compute_width_height_ratio(width, height)
        deviation = compute_deviation_from_one(wh_ratio)

        grain_records.append({
            "序号": idx,
            "实例ID": int(ins_id),
            "粒长": round(length, 4),
            "粒宽": round(width, 4),
            "粒高": round(height, 4),
            "长宽比": round(length / width, 3) if width > 0 else 0.0,
            "长高比": round(length / height, 3) if height > 0 else 0.0,
            "宽高比": round(wh_ratio, 3),
            "偏离度": round(deviation, 4),
        })

    deviations = [g["偏离度"] for g in grain_records]
    lengths = [g["粒长"] for g in grain_records]
    z_scores, mean_dev, std_dev = normalize_deviations_zscore(deviations)
    mean_length = float(np.mean(lengths)) if lengths else 0.0
    length_threshold = mean_length * min_length_ratio

    rows = []
    survival_map: dict[int, int] = {}
    for record, z_score in zip(grain_records, z_scores):
        is_firm, reason = is_grain_firm(
            z_score,
            record["粒长"],
            mean_length,
            deviation_z_threshold,
            min_length_ratio,
        )
        survival = 1 if is_firm else 0
        survival_map[record["实例ID"]] = survival

        row = {k: v for k, v in record.items() if k != "实例ID"}
        row["归一化偏离度"] = round(z_score, 3)
        row["圆润度"] = round(1.0 / (1.0 + record["偏离度"]), 3)
        row["粒长/均值"] = round(record["粒长"] / mean_length, 3) if mean_length > 0 else 0.0
        row["是否结实"] = "是" if is_firm else "否"
        row["判定说明"] = reason
        rows.append(row)

    detail_df = pd.DataFrame(rows)
    summary_df = build_summary_df(
        detail_df,
        mean_dev,
        std_dev,
        deviation_z_threshold,
        mean_length,
        min_length_ratio,
        length_threshold,
    )
    return detail_df, summary_df, survival_map


def build_summary_df(
    detail_df: pd.DataFrame,
    mean_deviation: float,
    std_deviation: float,
    deviation_z_threshold: float,
    mean_length: float,
    min_length_ratio: float,
    length_threshold: float,
) -> pd.DataFrame:
    """构建汇总表：总粒数、结实粒数、结实率。"""
    total = len(detail_df)
    firm_count = int((detail_df["是否结实"] == "是").sum())
    firm_rate = round(firm_count / total * 100, 2) if total else 0.0

    return pd.DataFrame([
        {"指标": "总粒数", "数值": total},
        {"指标": "结实粒数", "数值": firm_count},
        {"指标": "结实率(%)", "数值": firm_rate},
        {"指标": "偏离度均值", "数值": round(mean_deviation, 4)},
        {"指标": "偏离度标准差", "数值": round(std_deviation, 4)},
        {"指标": "归一化偏离阈值", "数值": deviation_z_threshold},
        {"指标": "粒长均值", "数值": round(mean_length, 4)},
        {"指标": "粒长比例阈值", "数值": min_length_ratio},
        {"指标": "粒长判定阈值", "数值": round(length_threshold, 4)},
        {"指标": "结实判定条件", "数值": f"Z-score<={deviation_z_threshold} 且 粒长>={round(length_threshold, 4)}"},
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
    """导出带 Survival_rate 的点云 TXT（结实=1，不结实=0）。"""
    txt_path = Path(txt_path)
    temp_path = txt_path.with_suffix(txt_path.suffix + ".tmp")

    with open(temp_path, "w", encoding="utf-8") as f:
        f.write("// X Y Z Ins Sem Survival_rate\n")
        f.write(f"// Overall_Survival_rate: {overall_survival_rate:.4f}\n")
        for i in range(len(points)):
            ins_id = int(instance_ids[i])
            survival = survival_map.get(ins_id, 0)
            f.write(
                f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                f"{ins_id} {int(sem[i])} {survival}\n"
            )

    atomic_replace(temp_path, txt_path, "文件")
    print(f"Survival TXT 已保存: {txt_path}")


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
    print(f"偏离度均值: {summary['偏离度均值']}")
    print(f"偏离度标准差: {summary['偏离度标准差']}")
    print(f"归一化偏离阈值: {summary['归一化偏离阈值']}")
    print(f"粒长均值: {summary['粒长均值']}")
    print(f"粒长比例阈值: {summary['粒长比例阈值']}")
    print(f"粒长判定阈值: {summary['粒长判定阈值']}")
    print(f"结实判定条件: {summary['结实判定条件']}")
    print("=" * 48)


def main():
    default_input = (
        r"E:\rice\hhy-0402-1\hhy-0402-1_psnppcuda.txt"
    )

    parser = argparse.ArgumentParser(description="穗粒尺寸测量与结实率评估，导出 Excel")
    parser.add_argument("--input_txt", type=str, default=default_input, help="输入点云 TXT 文件")
    parser.add_argument("--output_excel", type=str, default=None, help="输出 Excel 路径")
    parser.add_argument("--output_txt", type=str, default=None, help="输出 Survival_rate TXT 路径")
    parser.add_argument("--distance_percentile", type=float, default=95.0, help="散点过滤百分位阈值")
    parser.add_argument("--std_ratio", type=float, default=2.0, help="散点过滤标准差倍数")
    parser.add_argument(
        "--deviation_z_threshold",
        type=float,
        default=1.5,
        help="宽高比偏离度的 Z-score 阈值，超过则判定为非结实（默认 1.5）",
    )
    parser.add_argument(
        "--min_length_ratio",
        type=float,
        default=0.4,
        help="粒长低于全体粒长均值 × 该比例时判定为非结实（默认 0.4）",
    )
    args = parser.parse_args()

    input_path = Path(args.input_txt)
    output_excel, output_txt = resolve_output_paths(
        input_path,
        args.output_excel,
        args.output_txt,
    )

    points, instance_ids, sem = load_pointcloud_txt(str(input_path))
    detail_df, summary_df, survival_map = measure_all_grains(
        points,
        instance_ids,
        distance_percentile=args.distance_percentile,
        std_ratio=args.std_ratio,
        deviation_z_threshold=args.deviation_z_threshold,
        min_length_ratio=args.min_length_ratio,
    )

    summary = dict(zip(summary_df["指标"], summary_df["数值"]))
    overall_survival_rate = summary["结实率(%)"] / 100.0

    save_excel(detail_df, summary_df, str(output_excel))
    save_survival_txt(
        str(output_txt),
        points,
        instance_ids,
        sem,
        survival_map,
        overall_survival_rate,
    )
    print_summary(summary_df)


if __name__ == "__main__":
    main()
