一、整体流程
两步完成：

点云 TXT  →  compute_grain_dimensions.py  →  Survival TXT + Excel
                                                    ↓
                              render_grain_survival_video.py  →  MP4 视频
1
compute_grain_dimensions.py
测量每颗穗粒尺寸，判定饱满/不饱满，导出带 Survival_rate 的 TXT
2
render_grain_survival_video.py
根据 Survival TXT 生成三阶段展示视频
二、环境准备
pip install numpy opencv-python pillow
numpy：点云与矩阵计算
opencv-python：渲染点、写视频
pillow：绘制中文标签（避免视频里出现 ???）
Windows 上会自动使用 C:/Windows/Fonts/msyh.ttc（微软雅黑）。

三、步骤 1：生成 Survival TXT
命令

python compute_grain_dimensions.py `
  --input_txt "E:\rice\hhy-0402-1\hhy-0402-1_psnppcuda.txt" `
  --output_excel "fused_colored_pointcloud_final.xlsx" `
  --output_txt "fused_colored_pointcloud_final_survival.txt"

Survival TXT 格式：

// X Y Z Ins Sem Survival_rate
12.3 45.6 7.8 31 1 1
...


四、步骤 2：生成展示视频
最简命令（使用默认路径）
python render_grain_survival_video.py
默认：
输入：e:\cloud-nerf\fused_colored_pointcloud_final_survival.txt
输出：e:\cloud-nerf\grain_explosion_video.mp4
时长：30 秒，30 fps，1280×720
自定义命令示例
python render_grain_survival_video.py `
  --input_txt "fused_colored_pointcloud_final_survival.txt" `
  --output_video "grain_explosion_video.mp4" `
  --duration 30 `
  --fps 30 `
  --width 1280 `
  --height 720 `
  --phase1_ratio 0.25 `
  --phase2_ratio 0.25
全部参数
参数	默认	说明
--input_txt
fused_colored_pointcloud_final_survival.txt
Survival TXT
--output_video
grain_explosion_video.mp4
输出 MP4
