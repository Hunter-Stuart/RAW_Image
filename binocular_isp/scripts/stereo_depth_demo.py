#!/usr/bin/env python
"""
双目视差与深度 demo —— 3D 目标检测的几何地基

流程: KITTI 标定 → 极线校正 → SGBM 立体匹配 → 视差图 → Q矩阵深度图
      → 与 Scene Flow 官方视差真值对比（坏点率）→ 保存可视化

用法（本地, 用已有的 KITTI Scene Flow 数据）:
  python scripts/stereo_depth_demo.py \
      --left ../data_scene_flow/training/image_2/000000_10.png \
      --right ../data_scene_flow/training/image_3/000000_10.png \
      --width 1242 --height 375 \
      --gt-disp ../data_scene_flow/training/disp_noc_0/000000_10.png

输出: data/output/depth_demo/
  disparity.png     视差热力图
  depth.png         深度图（近=亮）
  disparity.npy     原始视差（供后续 3D 检测使用）
  pointcloud.ply    下采样点云（MeshLab 可打开）
  metrics.json      视差统计与坏点率（对 GT）
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from binocular.rectify import StereoRectifier, StereoCalibration
from config.camera_params import ConfigLoader


def build_rectifier(config_dir, image_size):
    loader = ConfigLoader(config_dir)
    lc = loader.load_camera_config('left', 'camera_left.yaml')
    rc = loader.load_camera_config('right', 'camera_right.yaml')
    sc = loader.load('stereo_calibration.yaml')
    ext = sc.get('stereo_extrinsics', {})
    calib = StereoCalibration(
        left_camera_matrix=loader.config_to_camera_matrix(lc),
        left_dist_coeffs=loader.config_to_dist_coeffs(lc),
        right_camera_matrix=loader.config_to_camera_matrix(rc),
        right_dist_coeffs=loader.config_to_dist_coeffs(rc),
        rotation_matrix=ConfigLoader.config_to_rotation(ext),
        translation_vector=ConfigLoader.config_to_translation(ext),
        image_size=image_size,
    )
    rectifier = StereoRectifier(calib)
    rectifier.compute_rectification_maps(new_size=image_size)
    return rectifier, calib


def main():
    parser = argparse.ArgumentParser(description='双目视差/深度 demo')
    parser.add_argument('--left', required=True)
    parser.add_argument('--right', required=True)
    parser.add_argument('--width', type=int, default=1242)
    parser.add_argument('--height', type=int, default=375)
    parser.add_argument('--config-dir', default=str(project_root / 'configs/kitti'))
    parser.add_argument('--gt-disp', default=None, help='Scene Flow 官方视差真值 PNG（可选）')
    parser.add_argument('--max-disp', type=int, default=128, help='SGBM 最大视差')
    args = parser.parse_args()

    out = project_root / 'data/output/depth_demo'
    out.mkdir(parents=True, exist_ok=True)

    rectifier, calib = build_rectifier(args.config_dir, (args.width, args.height))
    print(f'标定: 基线={np.linalg.norm(calib.translation_vector):.3f}m, fx={calib.left_camera_matrix[0,0]:.1f}')

    left = cv2.imread(args.left, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(args.right, cv2.IMREAD_GRAYSCALE)
    left_r, right_r = rectifier.rectify(left, right)

    # SGBM 立体匹配
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=args.max_disp, blockSize=5,
        P1=8 * 3 * 5 ** 2, P2=32 * 3 * 5 ** 2,
        disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=32)
    disp = sgbm.compute(left_r, right_r).astype(np.float32) / 16.0

    # Q 矩阵 → 深度。
    # 注意: OpenCV stereoRectify 的 T 约定为"左相机在右相机系中的位置"（与
    # stereoCalibrate 返回的"右相机在左相机系"方向相反），因此 Q[3,2]/Q[3,3]
    # 需取反，否则 reprojectImageTo3D 的深度为负。此修复已实测验证。
    Q = calib.Q.copy()
    Q[3, 2] *= -1
    Q[3, 3] *= -1
    depth = cv2.reprojectImageTo3D(disp, Q)[:, :, 2]
    valid = (disp > 0) & (depth > 0) & (depth < 100)

    # 视差热力图
    disp_vis = np.clip(disp, 0, args.max_disp)
    disp_col = cv2.applyColorMap((disp_vis / args.max_disp * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(out / 'disparity.png'), disp_col)

    # 深度图（近=亮，10m 内）
    depth_vis = np.clip(depth, 0, 50)
    depth_col = cv2.applyColorMap((depth_vis / 50 * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(out / 'depth.png'), depth_col)

    np.save(out / 'disparity.npy', disp)

    # 点云（下采样 1/4，PLY 格式，MeshLab 可看）
    pts = cv2.reprojectImageTo3D(disp, calib.Q)
    mask = valid & (np.arange(args.height)[:, None] % 4 == 0) & (np.arange(args.width)[None, :] % 4 == 0)
    p = pts[mask]
    colors = cv2.cvtColor(cv2.imread(args.left), cv2.COLOR_BGR2RGB).astype(np.float32)[mask]
    with open(out / 'pointcloud.ply', 'w') as f:
        f.write('ply\nformat ascii 1.0\nelement vertex %d\n' % len(p))
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for (x, y, z), (r, g, b) in zip(p, colors):
            f.write(f'{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n')

    # 统计
    metrics = {
        'baseline_m': float(np.linalg.norm(calib.translation_vector)),
        'valid_pixel_ratio': float(valid.mean()),
        'depth_median_m': float(np.median(depth[valid])),
        'disparity_mean_px': float(disp[valid].mean()),
    }

    # 与 GT 视差对比（Scene Flow disp_noc_0 为 uint16，值/256 = 像素）
    if args.gt_disp:
        gt = cv2.imread(args.gt_disp, cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
        if gt.shape == disp.shape:
            gt_valid = gt > 0
            both = valid & gt_valid
            err = np.abs(disp - gt)[both]
            metrics['gt_bad_pixel_ratio_3px'] = float((err > 3).mean())
            metrics['gt_mae_px'] = float(err.mean())
            metrics['gt_eval_pixels'] = int(both.sum())
            print(f'  vs GT: 坏点率(>3px)={metrics["gt_bad_pixel_ratio_3px"]:.3f}, '
                  f'MAE={metrics["gt_mae_px"]:.2f}px')

    json.dump(metrics, open(out / 'metrics.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'  有效像素占比: {metrics["valid_pixel_ratio"]:.2f}, 深度中位数: {metrics["depth_median_m"]:.1f}m')
    print(f'产出: {out}/disparity.png, depth.png, pointcloud.ply, disparity.npy, metrics.json')


if __name__ == '__main__':
    main()
