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
    compute_bbox_dimensions,
    compute_deviation_from_one,
    compute_width_height_ratio,
    filter_center_outliers,
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
    distance_percentile: float = 95.0,
    std_ratio: float = 2.0,
) -> dict[int, float]:
    """计算每个实例的圆润度（宽高比越接近 1 越高）。"""
    score_map: dict[int, float] = {}
    for ins_id in np.unique(instance_ids):
        inst_points = points[instance_ids == ins_id]
        filtered = filter_center_outliers(
            inst_points,
            distance_percentile=distance_percentile,
            std_ratio=std_ratio,
        )
        _length, width, height = compute_bbox_dimensions(filtered)
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


def select_firm_and_hollow_grains(
    instance_ids: np.ndarray,
    survival: np.ndarray,
    roundness_map: dict[int, float],
    n_each: int = 2,
) -> tuple[list[int], list[float], list[int], list[float]]:
    """实心=Survival_rate 1 中圆润度最高；空心=Survival_rate 0 中圆润度最低。"""
    surv_map = get_instance_survival(instance_ids, survival)

    firm_candidates = [(i, roundness_map[i]) for i in surv_map if surv_map[i] == 1 and i in roundness_map]
    hollow_candidates = [(i, roundness_map[i]) for i in surv_map if surv_map[i] == 0 and i in roundness_map]
    firm_candidates.sort(key=lambda x: x[1], reverse=True)
    hollow_candidates.sort(key=lambda x: x[1])

    if len(firm_candidates) < n_each:
        raise ValueError(f"饱满稻穗不足 {n_each} 颗（当前 {len(firm_candidates)} 颗）")
    if len(hollow_candidates) < n_each:
        all_sorted = sorted(roundness_map.items(), key=lambda x: x[1])
        hollow_candidates = all_sorted[:n_each]
        print(f"警告：不饱满稻穗不足 {n_each} 颗，改选圆润度最低的 {n_each} 颗")

    top_items = firm_candidates[:n_each]
    bottom_items = hollow_candidates[:n_each]
    top_ids = [int(i) for i, _ in top_items]
    top_scores = [float(v) for _, v in top_items]
    bottom_ids = [int(i) for i, _ in bottom_items]
    bottom_scores = [float(v) for _, v in bottom_items]

    print("选中的 2 颗饱满稻穗:")
    for ins_id, score in zip(top_ids, top_scores):
        print(f"  实例 #{ins_id}: 圆润度 = {score:.3f}")
    print("选中的 2 颗不饱满稻穗:")
    for ins_id, score in zip(bottom_ids, bottom_scores):
        print(f"  实例 #{ins_id}: 圆润度 = {score:.3f}")

    return top_ids, top_scores, bottom_ids, bottom_scores


def build_showcase_items(
    firm_ids: list[int],
    firm_scores: list[float],
    hollow_ids: list[int],
    hollow_scores: list[float],
) -> list[dict]:
    """按饱满度（圆润度）从高到低排序，用于从左到右展示。"""
    items = []
    for ins_id, score in zip(firm_ids, firm_scores):
        items.append({"id": int(ins_id), "score": float(score), "is_firm": True})
    for ins_id, score in zip(hollow_ids, hollow_scores):
        items.append({"id": int(ins_id), "score": float(score), "is_firm": False})
    items.sort(key=lambda x: x["score"], reverse=True)
    print("展示排列（饱满→植株左侧，不饱满→植株右侧）:")
    for idx, item in enumerate(items):
        label = "饱满" if item["is_firm"] else "不饱满"
        print(f"  位置{idx + 1}: #{item['id']} {label} 圆润度={item['score']:.3f}")
    return items


def compute_instance_centroids(points: np.ndarray, instance_ids: np.ndarray) -> dict[int, np.ndarray]:
    centroids: dict[int, np.ndarray] = {}
    for ins_id in np.unique(instance_ids):
        centroids[int(ins_id)] = points[instance_ids == ins_id].mean(axis=0)
    return centroids


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


def assign_grain_target_positions(showcase_items: list[dict], width: int, height: int) -> list[tuple[int, int]]:
    """饱满稻穗飞到植株左侧，不饱满稻穗飞到植株右侧。"""
    firm_slots = [
        (int(width * 0.10), int(height * 0.38)),
        (int(width * 0.10), int(height * 0.62)),
    ]
    hollow_slots = [
        (int(width * 0.90), int(height * 0.38)),
        (int(width * 0.90), int(height * 0.62)),
    ]
    firm_i = hollow_i = 0
    targets: list[tuple[int, int]] = []
    for item in showcase_items:
        if item["is_firm"]:
            targets.append(firm_slots[firm_i])
            firm_i += 1
        else:
            targets.append(hollow_slots[hollow_i])
            hollow_i += 1
    return targets


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
    label = "饱满" if is_firm else "不饱满"
    return f"#{ins_id} {label} 圆润度={score:.3f}"


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

def render_grain_showcase(
    canvas, grain_points, grain_colors, long_axis, center, spin_angle, view, width, height,
    tint, point_radius, target_screen_uv, move_progress, display_scale, proj_scale,
):
    """稻穗沿长轴旋转，并移动到画面上指定像素位置（视角中从左到右排开）。"""
    grain_center = grain_points.mean(axis=0)
    local = (grain_points - grain_center) @ rotation_matrix_axis(long_axis, spin_angle).T
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
    """植株保持原视角，4 颗稻穗飞出/展示（仅放大稻穗本身）。"""
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
        mask = instance_ids == ins_id
        u, v = render_grain_showcase(
            canvas, points[mask], colors[mask], long_axes[ins_id], center, spin_angle,
            view, width, height, tint, highlight_radius,
            target_positions[idx], move_t, grain_scale, plant_scale,
        )
        if show_labels and move_t > 0.85 and len(u) > 0:
            side = "左" if is_firm else "右"
            draw_label(
                canvas,
                f"#{ins_id} {grain_label_text(ins_id, is_firm, score)} [{side}]",
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
    distance_percentile: float = 95.0,
    std_ratio: float = 2.0,
):
    points, instance_ids, _sem, survival = load_survival_txt(input_txt)
    colors = build_instance_colors(instance_ids)
    center = points.mean(axis=0)

    roundness_map = compute_all_roundness_scores(
        points, instance_ids, distance_percentile=distance_percentile, std_ratio=std_ratio,
    )
    firm_ids, firm_scores, hollow_ids, hollow_scores = select_firm_and_hollow_grains(
        instance_ids, survival, roundness_map, n_each=2,
    )
    showcase_items = build_showcase_items(firm_ids, firm_scores, hollow_ids, hollow_scores)
    keep_ids = {item["id"] for item in showcase_items}
    keep_mask = np.isin(instance_ids, list(keep_ids))
    bg_indices = np.where(~keep_mask)[0]
    bg_indices = bg_indices[subsample_indices(len(bg_indices), max_render_points)]
    all_render_indices = subsample_indices(len(points), max_render_points)

    long_axes = compute_grain_long_axes(points, instance_ids, [item["id"] for item in showcase_items])
    global_long_axis = compute_long_axis(points[subsample_indices(len(points), min(30000, len(points)))])
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
                show_labels=True, title="饱满←植株→不饱满",
            )
        writer.write(canvas)
        if (frame_idx + 1) % 60 == 0 or frame_idx + 1 == total_frames:
            print(f"  进度: {frame_idx + 1}/{total_frames}")

    writer.release()
    print(f"视频已保存: {output_video}")


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="从 Survival TXT 生成稻穗爆炸展示视频")
    parser.add_argument("--input_txt", type=str, default=str(root / "fused_colored_pointcloud_final_survival.txt"))
    parser.add_argument("--output_video", type=str, default=str(root / "grain_explosion_video.mp4"))
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
