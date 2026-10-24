#!/usr/bin/env python
"""
KITTI 双目校正链路验证脚本

KITTI 图像是"已校正 + 已去畸变"的 RGB PNG，标定参数为
"无畸变 + 纯水平基线"，因此:
  - 校正前后图像应几乎不变（平均像素差异接近 0）
  - verify_rectification 的垂直误差应 ≈ 0（< 0.5px 达标）

这可以端到端验证: KITTI标定转换 → YAML加载 → StereoRectifier → 质量指标，
是相机到货前对"极线校正链路"最接近真实的检验。

用法:
  python scripts/test_rectify_kitti.py \
      --left <KITTI左图.png> --right <KITTI右图.png> \
      --width 1242 --height 375 --config-dir configs/kitti
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from binocular.rectify import StereoRectifier, StereoCalibration
from config.camera_params import ConfigLoader


def main():
    parser = argparse.ArgumentParser(description='KITTI 双目校正链路验证')
    parser.add_argument('--left', required=True, help='KITTI 左图 (cam2) PNG')
    parser.add_argument('--right', required=True, help='KITTI 右图 (cam3) PNG')
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--config-dir', default='configs/kitti', help='转换后的YAML所在目录')
    parser.add_argument('--output', default='data/output')
    args = parser.parse_args()

    loader = ConfigLoader(args.config_dir)
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
        image_size=(args.width, args.height),
    )
    rectifier = StereoRectifier(calib)
    rectifier.compute_rectification_maps()
    print(f'标定: T={calib.translation_vector.flatten()}, 基线={float(np.linalg.norm(calib.translation_vector)):.3f}m')

    left = cv2.imread(args.left)
    right = cv2.imread(args.right)
    if left is None or right is None:
        raise SystemExit('图像读取失败，请检查路径')

    left_r, right_r = rectifier.rectify(left, right)
    res = rectifier.verify_rectification(left_r, right_r)
    print(f'匹配对数: {res["num_matches"]}')
    print(f'平均垂直误差: {res["mean_vertical_error_px"]:.3f} px (要求 <0.5)')
    print(f'最大垂直误差: {res["max_vertical_error_px"]:.3f} px')
    print(f'通过0.5px阈值: {res["pass_THRESHOLD_0_5px"]}')

    # KITTI 已校正：校正前后差异应很小（边缘可能有微小裁剪差异）
    diff_l = float(np.abs(left.astype(np.float32) - left_r.astype(np.float32)).mean())
    diff_r = float(np.abs(right.astype(np.float32) - right_r.astype(np.float32)).mean())
    print(f'校正前后平均像素差异: 左={diff_l:.2f} 右={diff_r:.2f} (应接近0)')

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / 'kitti_rect_left.png'), left_r)
    cv2.imwrite(str(out / 'kitti_rect_right.png'), right_r)
    cv2.imwrite(str(out / 'kitti_epipolar.png'), StereoRectifier.draw_epipolar_lines(left_r, right_r))
    print(f'结果已保存: {out}/kitti_rect_left.png, kitti_rect_right.png, kitti_epipolar.png')


if __name__ == '__main__':
    main()
