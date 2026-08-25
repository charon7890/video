#!/usr/bin/env python3
"""
从 compute_grain_dimensions 导出的 Survival TXT 生成稻穗展示视频。

- 第一阶段：缩小视角，整株点云沿长轴旋转
- 第二阶段：4 颗精选稻穗从植株中飞出（饱满→左，不饱满→右），飞行中仅放大稻穗
- 第三阶段：植株保持原视角，4 颗稻穗在两侧沿长轴旋转展示
"""

import argparse
import colorsys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compute_grain_dimensions import (
    compute_deviation_from_one,
    compute_ellipsoid_dimensions,
    compute_width_height_ratio,
    fit_percentile_ellipsoid,
)


def load_survival_txt(txt_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载带 Survival_rate 的点云 TXT。"""
    print(f"正在加载点云: {txt_path}")
    data = np.loadtxt(txt_path, comments="//")
    points = data[:, :3]
    instance_ids = data[:, 3].astype(np.int32)
    sem = data[:, 4].astype(np.int32) if data.shape[1] > 4 else np.zeros(len(points), dtype=np.int32)
    survival = data[:, 5].astype(np.int32) if data.shape[1] > 5 else np.zeros(len(points), dtype=np.int32)
    print(f"已加载 {len(points)} 个点，{len(np.unique(instance_ids))} 个实例")
    return points, instance_ids, sem, survival


def compute_all_roundness_scores(
    points: np.ndarray,
    instance_ids: np.ndarray,
) -> dict[int, float]:
    """椭球拟合后计算圆润度（宽高比越接近 1 越高），不剔除离散点。"""
    score_map: dict[int, float] = {}
    for ins_id in np.unique(instance_ids):
        inst_points = points[instance_ids == ins_id]
        _length, width, height = compute_ellipsoid_dimensions(inst_points)
        wh_ratio = compute_width_height_ratio(width, height)
        deviation = compute_deviation_from_one(wh_ratio)
        score_map[int(ins_id)] = 1.0 / (1.0 + deviation)
    return score_map


def get_instance_survival(instance_ids: np.ndarray, survival: np.ndarray) -> dict[int, int]:
    surv_map: dict[int, int] = {}
    for ins_id in np.unique(instance_ids):
        mask = instance_ids == ins_id
        surv_map[int(ins_id)] = int(survival[mask][0])
    return surv_map


def compute_axis_extent(
    points: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
) -> tuple[float, float]:
    """全体点沿长轴投影的 [min, max]。"""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    all_proj = (points - center) @ axis
    return float(all_proj.min()), float(all_proj.max())


def count_instances_near_tips(
    heights: dict[int, float],
    h_min: float,
    h_max: float,
    tip_ratio: float = 0.20,
) -> tuple[list[int], list[int], float]:
    """
    统计靠近长轴两端「端处」的实例。
    端处 = 从该端起 tip_ratio * 轴长 的区间。
    返回 (高位端实例列表, 低位端实例列表, 端区长度)。
    """
    span = max(h_max - h_min, 1e-8)
    tip_len = span * tip_ratio
    high_tip_ids = [i for i, h in heights.items() if h >= h_max - tip_len]
    low_tip_ids = [i for i, h in heights.items() if h <= h_min + tip_len]
    return high_tip_ids, low_tip_ids, tip_len


def orient_long_axis_dense_end_up(
    points: np.ndarray,
    instance_ids: np.ndarray,
    axis: np.ndarray,
    tip_ratio: float = 0.20,
) -> np.ndarray:
    """
    比较长轴两端「端处」附近的实例数，实例更多的一端为上端；
    翻转长轴使该端指向 +axis（视野上方）。
    """
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    center = points.mean(axis=0)
    h_min, h_max = compute_axis_extent(points, center, axis)
    centroids = compute_instance_centroids(points, instance_ids)
    heights = {
        int(ins_id): float(np.dot(centroids[ins_id] - center, axis))
        for ins_id in centroids
    }
    high_tip_ids, low_tip_ids, tip_len = count_instances_near_tips(
        heights, h_min, h_max, tip_ratio=tip_ratio,
    )
    high_count = len(high_tip_ids)
    low_count = len(low_tip_ids)

    if low_count > high_count:
        axis = -axis
        print(
            f"视角定向: 端处实例 高位端 {high_count} / 低位端 {low_count} "
            f"（端区长度={tip_len:.4f}），穗密端在低位，已翻转长轴使上端朝向视野上方"
        )
    else:
        print(
            f"视角定向: 端处实例 高位端 {high_count} / 低位端 {low_count} "
            f"（端区长度={tip_len:.4f}），穗密端在高位，长轴已朝向视野上方"
        )
    return axis


def select_firm_and_hollow_grains(
    points: np.ndarray,
    instance_ids: np.ndarray,
    survival: np.ndarray,
    roundness_map: dict[int, float],
    plant_long_axis: np.ndarray,
    n_each: int = 2,
    tip_ratio: float = 0.20,
) -> tuple[list[int], list[float], list[int], list[float], list[str], list[str]]:
    """
    左侧结实 / 右侧非结实：均在穗密上端处选取靠近端处的 n_each 颗。
    """
    surv_map = get_instance_survival(instance_ids, survival)
    center = points.mean(axis=0)
    axis = plant_long_axis / (np.linalg.norm(plant_long_axis) + 1e-12)
    centroids = compute_instance_centroids(points, instance_ids)

    heights: dict[int, float] = {
        int(ins_id): float(np.dot(centroids[ins_id] - center, axis))
        for ins_id in centroids
    }
    h_min, h_max = compute_axis_extent(points, center, axis)
    high_tip_ids, low_tip_ids, tip_len = count_instances_near_tips(
        heights, h_min, h_max, tip_ratio=tip_ratio,
    )
    tip_ids = high_tip_ids
    tip_score = {i: h - (h_max - tip_len) for i, h in heights.items()}

    print(
        f"上端判定: 高位端处（已定向为视野上方） | "
        f"端区=[{h_max - tip_len:.4f}, {h_max:.4f}], "
        f"上端处 {len(high_tip_ids)} 粒, 下端处 {len(low_tip_ids)} 粒"
    )

    firm_all = [i for i in surv_map if surv_map[i] == 1 and i in roundness_map]
    hollow_all = [i for i in surv_map if surv_map[i] == 0 and i in roundness_map]

    tip_id_set = set(tip_ids)
    firm_pool = [i for i in firm_all if i in tip_id_set]
    hollow_pool = [i for i in hollow_all if i in tip_id_set]
    if len(firm_pool) < n_each:
        print(f"警告：上端处结实穗仅 {len(firm_pool)} 颗，改为在全部结实穗中按靠近上端选取")
        firm_pool = firm_all
    if len(hollow_pool) < n_each:
        print(f"警告：上端处非结实穗仅 {len(hollow_pool)} 颗，改为在全部非结实穗中按靠近上端选取")
        hollow_pool = hollow_all
    if len(firm_pool) < n_each:
        raise ValueError(f"结实稻穗不足 {n_each} 颗（当前 {len(firm_pool)} 颗）")
    if len(hollow_pool) < n_each:
        raise ValueError(
            f"非结实稻穗不足 {n_each} 颗（当前 {len(hollow_pool)} 颗），"
            f"右边必须先从非结实中选取"
        )

    def pick_near_tip(pool: list[int]) -> tuple[list[int], list[float], list[str]]:
        ranked = sorted(pool, key=lambda i: tip_score[i], reverse=True)
        chosen = [int(i) for i in ranked[:n_each]]
        ids = sorted(chosen, key=lambda i: heights[i], reverse=True)
        scores = [float(roundness_map[i]) for i in ids]
        ends = ["top", "bottom"] if len(ids) >= 2 else ["top"] * len(ids)
        return ids, scores, ends

    firm_ids, firm_scores, firm_ends = pick_near_tip(firm_pool)
    hollow_ids, hollow_scores, hollow_ends = pick_near_tip(hollow_pool)

    print("选中的 2 颗结实稻穗（靠近穗密上端处）:")
    for ins_id, score, end in zip(firm_ids, firm_scores, firm_ends):
        print(
            f"  实例 #{ins_id} [画面{end}]: 圆润度={score:.3f}, "
            f"轴向高度={heights[ins_id]:.4f}"
        )
    print("选中的 2 颗非结实稻穗（靠近穗密上端处）:")
    for ins_id, score, end in zip(hollow_ids, hollow_scores, hollow_ends):
        print(
            f"  实例 #{ins_id} [画面{end}]: 圆润度={score:.3f}, "
            f"轴向高度={heights[ins_id]:.4f}"
        )

    return firm_ids, firm_scores, hollow_ids, hollow_scores, firm_ends, hollow_ends


def build_showcase_items(
    firm_ids: list[int],
    firm_scores: list[float],
    hollow_ids: list[int],
    hollow_scores: list[float],
    firm_ends: list[str],
    hollow_ends: list[str],
) -> list[dict]:
    """左侧结实、右侧非结实；每侧按画面上→下排列。"""
    items = []
    for ins_id, score, end in zip(firm_ids, firm_scores, firm_ends):
        items.append({"id": int(ins_id), "score": float(score), "is_firm": True, "end": end})
    for ins_id, score, end in zip(hollow_ids, hollow_scores, hollow_ends):
        items.append({"id": int(ins_id), "score": float(score), "is_firm": False, "end": end})

    def sort_key(item: dict) -> tuple:
        side = 0 if item["is_firm"] else 1
        vert = 0 if item.get("end") == "top" else 1
        return (side, vert)

    items.sort(key=sort_key)
    print("展示排列（左=结实靠近上端，右=非结实靠近上端）:")
    for item in items:
        label = "结实" if item["is_firm"] else "非结实"
        side = "左" if item["is_firm"] else "右"
        end = "上" if item.get("end") == "top" else "下"
        print(f"  {side}{end}: #{item['id']} {label}")
    return items


def compute_instance_centroids(points: np.ndarray, instance_ids: np.ndarray) -> dict[int, np.ndarray]:
    centroids: dict[int, np.ndarray] = {}
    for ins_id in np.unique(instance_ids):
        centroids[int(ins_id)] = points[instance_ids == ins_id].mean(axis=0)
    return centroids


def assign_grain_target_positions(showcase_items: list[dict], width: int, height: int) -> list[tuple[int, int]]:
    """饱满→左（上/下），不饱满→右（上/下）。"""
    slot_map = {
        ("firm", "top"): (int(width * 0.10), int(height * 0.32)),
        ("firm", "bottom"): (int(width * 0.10), int(height * 0.68)),
        ("hollow", "top"): (int(width * 0.90), int(height * 0.32)),
        ("hollow", "bottom"): (int(width * 0.90), int(height * 0.68)),
    }
    targets: list[tuple[int, int]] = []
    for item in showcase_items:
        key = ("firm" if item["is_firm"] else "hollow", item.get("end", "top"))
        targets.append(slot_map[key])
    return targets


def generate_distinct_colors(n: int) -> np.ndarray:
    colors = []
    golden_ratio = 0.618033988749895
    for i in range(n):
        hue = (i * golden_ratio) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.95 if i % 2 == 0 else 0.75)
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    return np.asarray(colors, dtype=np.uint8)


def build_instance_colors(instance_ids: np.ndarray) -> np.ndarray:
    unique_ids = np.sort(np.unique(instance_ids))
    id_to_idx = {int(ins_id): idx for idx, ins_id in enumerate(unique_ids)}
    palette = generate_distinct_colors(len(unique_ids))
    colors = np.zeros((len(instance_ids), 3), dtype=np.uint8)
    for i, ins_id in enumerate(instance_ids):
        colors[i] = palette[id_to_idx[int(ins_id)]]
    return colors


def rotation_matrix_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ], dtype=np.float64)



def compute_long_axis(grain_points: np.ndarray) -> np.ndarray:
    if len(grain_points) < 3:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    centered = grain_points - grain_points.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered.T))
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    return axis / (np.linalg.norm(axis) + 1e-12)


def compute_grain_long_axes(
    points: np.ndarray,
    instance_ids: np.ndarray,
    grain_ids: list[int],
) -> dict[int, np.ndarray]:
    return {int(ins_id): compute_long_axis(points[instance_ids == ins_id]) for ins_id in grain_ids}


def get_view_matrix_perpendicular_to_axis(axis: np.ndarray) -> np.ndarray:
    """
    构建观察矩阵：视线方向与长轴垂直，长轴对齐屏幕竖直方向。
    行向量依次为 [右, 上, 视向]，投影时 u←右、v←-上、depth←视向。
    """
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(axis, ref))) > 0.92:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    view_dir = np.cross(axis, ref)
    view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-12)
    right = np.cross(view_dir, axis)
    right = right / (np.linalg.norm(right) + 1e-12)
    up = axis
    return np.stack([right, up, view_dir], axis=0)


def ease_in_out(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


def subsample_indices(n: int, max_points: int, seed: int = 42) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def world_to_view_offset(world_offset: np.ndarray, view: np.ndarray) -> np.ndarray:
    return world_offset @ view.T


def view_to_world_offset(view_offset: np.ndarray, view: np.ndarray) -> np.ndarray:
    return view_offset @ view


def compute_fit_scale(
    local_points: np.ndarray,
    view: np.ndarray,
    width: int,
    height: int,
    margin: float = 0.88,
) -> float:
    """根据点云投影范围自动计算缩放，使整株植株完整落入画面。"""
    if len(local_points) == 0:
        return min(width, height) * 0.15
    proj = local_points @ view.T
    u_extent = float(np.ptp(proj[:, 0]))
    v_extent = float(np.ptp(proj[:, 1]))
    if u_extent < 1e-6 or v_extent < 1e-6:
        return min(width, height) * 0.15
    scale_u = (width * margin) / u_extent
    scale_v = (height * margin) / v_extent
    return min(scale_u, scale_v)


def project_points(
    points: np.ndarray,
    view_matrix: np.ndarray,
    width: int,
    height: int,
    scale: float,
    pan_u: float = 0.0,
    pan_v: float = 0.0,
):
    centered = points @ view_matrix.T
    u = (centered[:, 0] * scale + width * 0.5 + pan_u).astype(np.int32)
    v = (-centered[:, 1] * scale + height * 0.5 + pan_v).astype(np.int32)
    return u, v, centered[:, 2]


def draw_points(canvas, u, v, depth, colors, point_radius, alpha=1.0):
    if len(u) == 0:
        return
    h, w = canvas.shape[:2]
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, depth, colors = u[valid], v[valid], depth[valid], colors[valid]
    if len(u) == 0:
        return
    order = np.argsort(depth)
    u, v, colors = u[order], v[order], colors[order]
    if point_radius <= 1:
        if alpha < 1.0:
            canvas[v, u] = (
                canvas[v, u].astype(np.float32) * (1 - alpha) + colors.astype(np.float32) * alpha
            ).astype(np.uint8)
        else:
            canvas[v, u] = colors
        return
    for idx in range(len(u)):
        cv2.circle(canvas, (int(u[idx]), int(v[idx])), point_radius, colors[idx].tolist(), -1, cv2.LINE_AA)


@lru_cache(maxsize=8)
def _load_chinese_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ):
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def draw_label(canvas: np.ndarray, text: str, position: tuple[int, int], color: tuple[int, int, int], font_size: int = 22) -> None:
    """使用 PIL 绘制中文标签（OpenCV putText 不支持中文会显示问号）。"""
    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font = _load_chinese_font(font_size)
    x, y = position
    rgb = (int(color[2]), int(color[1]), int(color[0]))
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, 1), (-1, 1), (1, -1)):
        draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=rgb)
    canvas[:] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def grain_label_text(ins_id: int, is_firm: bool, score: float) -> str:
    # 画面不展示圆润度，仅保留结实/非结实
    _ = score
    label = "结实" if is_firm else "非结实"
    return f"#{ins_id} {label}"


def render_frame_full_plant_rotation(
    points, colors, render_indices, center, global_long_axis, angle, width, height, point_radius, plant_scale,
):
    """整株点云旋转（缩小视角，可见完整植株）。"""
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    local = (points[render_indices] - center) @ rotation_matrix_axis(global_long_axis, angle).T
    view = get_view_matrix_perpendicular_to_axis(global_long_axis)
    u, v, depth = project_points(local, view, width, height, plant_scale)
    draw_points(canvas, u, v, depth, colors[render_indices], point_radius, alpha=1.0)
    draw_label(canvas, "整株稻穗点云", (24, 40), (220, 220, 220))
    return canvas


def render_plant_background(
    canvas: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    bg_indices: np.ndarray,
    center: np.ndarray,
    view: np.ndarray,
    plant_scale: float,
    width: int,
    height: int,
    point_radius: int,
) -> None:
    """绘制整株背景（不含飞出的 4 颗稻穗），视角缩放不变。"""
    if len(bg_indices) == 0:
        return
    local = points[bg_indices] - center
    u, v, depth = project_points(local, view, width, height, plant_scale)
    draw_points(canvas, u, v, depth, colors[bg_indices], point_radius, alpha=1.0)

def compute_obb_params(
    points: np.ndarray,
    padding: float = 0.05,
    flat_scale: float = 1.0,
    low_percentile: float = 2.5,
    high_percentile: float = 97.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    与测量相同的椭球拟合，再用轴向方框框住该椭球。
    椭球 = fit_percentile_ellipsoid；方框 = 椭球轴向包络（可加 padding / 压扁最短轴）。
    """
    if len(points) < 3:
        eye = np.eye(3, dtype=np.float64)
        zeros = np.zeros(3, dtype=np.float64)
        return zeros, eye, zeros, zeros

    fitted = fit_percentile_ellipsoid(
        points,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
    )
    if fitted is None:
        eye = np.eye(3, dtype=np.float64)
        zeros = np.zeros(3, dtype=np.float64)
        return zeros, eye, zeros, zeros

    center, axes, mins, maxs = fitted
    extents = np.maximum(maxs - mins, 1e-6)
    mins = mins - extents * padding
    maxs = maxs + extents * padding

    flat = float(np.clip(flat_scale, 0.15, 1.0))
    if flat < 1.0:
        # 最短轴为 PCA 第 3 轴（特征值最小）
        mid = 0.5 * (mins[2] + maxs[2])
        half = 0.5 * (maxs[2] - mins[2]) * flat
        mins[2] = mid - half
        maxs[2] = mid + half

    return center, axes, mins, maxs


def compute_obb_corners(
    points: np.ndarray,
    padding: float = 0.05,
    flat_scale: float = 1.0,
) -> np.ndarray:
    """椭球外接方框的 8 个角点（相对质心）。"""
    center, axes, mins, maxs = compute_obb_params(points, padding, flat_scale)
    if len(points) < 3:
        return np.zeros((8, 3), dtype=np.float64)

    local_corners = np.array([
        [mins[0], mins[1], mins[2]],
        [maxs[0], mins[1], mins[2]],
        [maxs[0], maxs[1], mins[2]],
        [mins[0], maxs[1], mins[2]],
        [mins[0], mins[1], maxs[2]],
        [maxs[0], mins[1], maxs[2]],
        [maxs[0], maxs[1], maxs[2]],
        [mins[0], maxs[1], maxs[2]],
    ], dtype=np.float64)
    return local_corners @ axes.T


def clip_points_to_obb(
    points: np.ndarray,
    colors: np.ndarray,
    padding: float = 0.05,
    flat_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """只保留椭球外接方框内的点，去掉框外离散点。"""
    if len(points) < 3:
        return points, colors

    center, axes, mins, maxs = compute_obb_params(points, padding=padding, flat_scale=flat_scale)
    local = (points - center) @ axes
    inside = (
        (local[:, 0] >= mins[0]) & (local[:, 0] <= maxs[0])
        & (local[:, 1] >= mins[1]) & (local[:, 1] <= maxs[1])
        & (local[:, 2] >= mins[2]) & (local[:, 2] <= maxs[2])
    )
    if not np.any(inside):
        return points, colors
    return points[inside], colors[inside]


_AABB_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def draw_bbox_wireframe(
    canvas: np.ndarray,
    corners_world: np.ndarray,
    view: np.ndarray,
    width: int,
    height: int,
    proj_scale: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """将 3D 长方体 8 角点投影到画面并绘制 12 条边。"""
    u, v, _ = project_points(corners_world, view, width, height, proj_scale)
    h, w = canvas.shape[:2]
    for i, j in _AABB_EDGES:
        p1 = (int(u[i]), int(v[i]))
        p2 = (int(u[j]), int(v[j]))
        if (
            (p1[0] < -50 and p2[0] < -50)
            or (p1[0] > w + 50 and p2[0] > w + 50)
            or (p1[1] < -50 and p2[1] < -50)
            or (p1[1] > h + 50 and p2[1] > h + 50)
        ):
            continue
        cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)


def render_grain_showcase(
    canvas, grain_points, grain_colors, long_axis, center, spin_angle, view, width, height,
    tint, point_radius, target_screen_uv, move_progress, display_scale, proj_scale,
    draw_bbox: bool = True, bbox_bgr: tuple[int, int, int] | None = None,
    bbox_flat_scale: float = 1.0, clip_outside_bbox: bool = True,
):
    """稻穗沿长轴旋转，并移动到画面指定位置；方形框贴合椭球分位包络，并裁掉框外点。"""
    # 框与裁剪都基于完整点云的 PCA 百分位包络（不含极值飞点）
    bbox_source_points = grain_points
    if clip_outside_bbox:
        grain_points, grain_colors = clip_points_to_obb(
            grain_points, grain_colors, padding=0.05, flat_scale=bbox_flat_scale,
        )

    grain_center = bbox_source_points.mean(axis=0)
    spin_rot = rotation_matrix_axis(long_axis, spin_angle)
    local = (grain_points - grain_center) @ spin_rot.T
    local = local * display_scale

    start_view = world_to_view_offset(grain_center - center, view)
    target_u, target_v = target_screen_uv
    target_view = start_view.copy()
    target_view[0] = (target_u - width * 0.5) / proj_scale
    target_view[1] = -(target_v - height * 0.5) / proj_scale

    current_view = start_view * (1.0 - move_progress) + target_view * move_progress
    center_offset = view_to_world_offset(current_view, view)
    world = local + center_offset

    u, v, depth = project_points(world, view, width, height, proj_scale)
    tinted = np.clip(grain_colors.astype(np.float32) * 0.35 + tint * 0.65, 0, 255).astype(np.uint8)
    draw_points(canvas, u, v, depth, tinted, point_radius, alpha=1.0)

    if draw_bbox and move_progress > 0.15:
        corners_local = (
            compute_obb_corners(bbox_source_points, padding=0.05, flat_scale=bbox_flat_scale)
            @ spin_rot.T
            * display_scale
        )
        corners_world = corners_local + center_offset
        color = bbox_bgr if bbox_bgr is not None else (int(tint[2]), int(tint[1]), int(tint[0]))
        draw_bbox_wireframe(
            canvas, corners_world, view, width, height, proj_scale, color, thickness=2,
        )

    return u, v


def render_frame_grains_scene(
    points,
    colors,
    instance_ids,
    bg_indices,
    center,
    showcase_items,
    target_positions,
    long_axes,
    global_long_axis,
    fly_progress,
    spin_angle,
    width,
    height,
    point_radius,
    highlight_radius,
    plant_scale,
    show_labels: bool = False,
    title: str | None = None,
):
    """植株保持原视角，4 颗稻穗飞出/展示（仅放大稻穗本身），并用长方体框选。"""
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    view = get_view_matrix_perpendicular_to_axis(global_long_axis)
    render_plant_background(
        canvas, points, colors, bg_indices, center, view, plant_scale, width, height, point_radius,
    )

    move_t = ease_in_out(float(np.clip(fly_progress, 0.0, 1.0)))
    grain_scale = 1.0 + move_t * 2.8

    for idx, item in enumerate(showcase_items):
        ins_id = item["id"]
        score = item["score"]
        is_firm = item["is_firm"]
        tint = np.array([80, 220, 120], dtype=np.float32) if is_firm else np.array([240, 90, 70], dtype=np.float32)
        bbox_bgr = (80, 230, 130) if is_firm else (90, 120, 255)  # BGR: 绿 / 红
        # 不饱满：最短边压到 50%，并去掉长方体外部的点
        bbox_flat = 1.0 if is_firm else 0.50
        mask = instance_ids == ins_id
        u, v = render_grain_showcase(
            canvas, points[mask], colors[mask], long_axes[ins_id], center, spin_angle,
            view, width, height, tint, highlight_radius,
            target_positions[idx], move_t, grain_scale, plant_scale,
            draw_bbox=True, bbox_bgr=bbox_bgr, bbox_flat_scale=bbox_flat,
            clip_outside_bbox=True,
        )
        if show_labels and move_t > 0.85 and len(u) > 0:
            side = "左上" if is_firm and item.get("end") == "top" else (
                "左下" if is_firm else (
                    "右上" if item.get("end") == "top" else "右下"
                )
            )
            draw_label(
                canvas,
                f"{grain_label_text(ins_id, is_firm, score)} [{side}]",
                (int(np.clip(u.mean() - 110, 24, width - 380)), int(np.clip(v.max() + 10, 30, height - 50))),
                (80, 230, 130) if is_firm else (255, 120, 90),
                font_size=18,
            )

    if title:
        draw_label(canvas, title, (24, 40), (220, 220, 220))
    return canvas


def render_video(
    input_txt: str,
    output_video: str,
    fps: int = 30,
    duration_sec: float = 30.0,
    phase1_ratio: float = 0.25,
    phase2_ratio: float = 0.25,
    width: int = 1280,
    height: int = 720,
    max_render_points: int = 100000,
    point_radius: int = 1,
    highlight_point_radius: int = 2,
):
    points, instance_ids, _sem, survival = load_survival_txt(input_txt)
    colors = build_instance_colors(instance_ids)
    center = points.mean(axis=0)

    roundness_map = compute_all_roundness_scores(points, instance_ids)
    global_long_axis = compute_long_axis(points[subsample_indices(len(points), min(30000, len(points)))])
    # 保证穗密上端朝向 +axis，从而在视野中位于屏幕上方
    global_long_axis = orient_long_axis_dense_end_up(points, instance_ids, global_long_axis)
    firm_ids, firm_scores, hollow_ids, hollow_scores, firm_ends, hollow_ends = select_firm_and_hollow_grains(
        points, instance_ids, survival, roundness_map, global_long_axis, n_each=2,
    )
    showcase_items = build_showcase_items(
        firm_ids, firm_scores, hollow_ids, hollow_scores, firm_ends, hollow_ends,
    )
    keep_ids = {item["id"] for item in showcase_items}
    keep_mask = np.isin(instance_ids, list(keep_ids))
    bg_indices = np.where(~keep_mask)[0]
    bg_indices = bg_indices[subsample_indices(len(bg_indices), max_render_points)]
    all_render_indices = subsample_indices(len(points), max_render_points)

    long_axes = compute_grain_long_axes(points, instance_ids, [item["id"] for item in showcase_items])
    view = get_view_matrix_perpendicular_to_axis(global_long_axis)
    plant_scale = compute_fit_scale(points - center, view, width, height, margin=0.90)
    target_positions = assign_grain_target_positions(showcase_items, width, height)
    print(f"  整株缩放: {plant_scale:.1f}（全程不变，仅放大飞出稻穗）")

    total_frames = int(fps * duration_sec)
    phase1_frames = max(1, int(total_frames * phase1_ratio))
    phase2_frames = max(1, int(total_frames * phase2_ratio))
    phase3_frames = max(1, total_frames - phase1_frames - phase2_frames)

    writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频: {output_video}")

    print(
        f"开始渲染 {total_frames} 帧\n"
        f"  阶段1 整株旋转: {phase1_frames} 帧\n"
        f"  阶段2 稻穗飞出: {phase2_frames} 帧\n"
        f"  阶段3 两侧展示: {phase3_frames} 帧\n"
        f"  输出: {output_video}"
    )

    for frame_idx in range(total_frames):
        if frame_idx < phase1_frames:
            angle = (frame_idx / max(1, phase1_frames - 1)) * np.pi * 2.0
            canvas = render_frame_full_plant_rotation(
                points, colors, all_render_indices, center, global_long_axis, angle,
                width, height, point_radius, plant_scale,
            )
        elif frame_idx < phase1_frames + phase2_frames:
            local_idx = frame_idx - phase1_frames
            progress = local_idx / max(1, phase2_frames - 1)
            canvas = render_frame_grains_scene(
                points, colors, instance_ids, bg_indices, center,
                showcase_items, target_positions, long_axes, global_long_axis,
                fly_progress=progress, spin_angle=progress * np.pi * 2.0,
                width=width, height=height, point_radius=point_radius,
                highlight_radius=highlight_point_radius, plant_scale=plant_scale,
                show_labels=False, title="精选稻穗飞出植株",
            )
        else:
            local_idx = frame_idx - phase1_frames - phase2_frames
            spin = (local_idx / max(1, phase3_frames - 1)) * np.pi * 4.0
            canvas = render_frame_grains_scene(
                points, colors, instance_ids, bg_indices, center,
                showcase_items, target_positions, long_axes, global_long_axis,
                fly_progress=1.0, spin_angle=spin,
                width=width, height=height, point_radius=point_radius,
                highlight_radius=highlight_point_radius, plant_scale=plant_scale,
                show_labels=False, title="结实←植株→非结实",
            )
        writer.write(canvas)
        if (frame_idx + 1) % 60 == 0 or frame_idx + 1 == total_frames:
            print(f"  进度: {frame_idx + 1}/{total_frames}")

    writer.release()
    print(f"视频已保存: {output_video}")


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="从 Survival TXT 生成稻穗爆炸展示视频")
    parser.add_argument("--input_txt", type=str, default=str(r"E:\rice\ply\013\fused_colored_pointcloud_final_survival.txt"))
    parser.add_argument("--output_video", type=str, default=str(r"E:\rice\ply\013\fused_colored_pointcloud_final_survival.mp4"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--phase1_ratio", type=float, default=0.25, help="整株旋转占比")
    parser.add_argument("--phase2_ratio", type=float, default=0.25, help="稻穗飞出占比")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max_render_points", type=int, default=100000)
    parser.add_argument("--point_radius", type=int, default=1)
    parser.add_argument("--highlight_point_radius", type=int, default=2)
    args = parser.parse_args()

    render_video(
        input_txt=args.input_txt,
        output_video=args.output_video,
        fps=args.fps,
        duration_sec=args.duration,
        phase1_ratio=args.phase1_ratio,
        phase2_ratio=args.phase2_ratio,
        width=args.width,
        height=args.height,
        max_render_points=args.max_render_points,
        point_radius=args.point_radius,
        highlight_point_radius=args.highlight_point_radius,
    )


if __name__ == "__main__":
    main()
