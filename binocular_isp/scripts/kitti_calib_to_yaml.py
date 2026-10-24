#!/usr/bin/env python
"""
KITTI 标定文件 → 本项目 YAML 配置转换器

将 KITTI 数据集的 calib_cam_to_cam.txt 转换为:
  camera_left.yaml / camera_right.yaml / stereo_calibration.yaml

KITTI 格式说明:
  P_rect_02 = 左彩色相机(cam2)校正后投影矩阵 3x4, 前 3 列为内参 K
  P_rect_03 = 右彩色相机(cam3)校正后投影矩阵 3x4
  KITTI 校正图已去畸变 → 畸变系数为 0, R 为单位阵
  基线 b = -P_rect_03[0,3] / fx ≈ 0.54 m

用法:
  python scripts/kitti_calib_to_yaml.py <calib_cam_to_cam.txt> \
      --width 1242 --height 375 --out configs/kitti
"""

import argparse
from pathlib import Path

import numpy as np
import yaml


def parse_kitti_calib(calib_path: str) -> dict:
    data = {}
    with open(calib_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, vals = line.split(':', 1)
            try:
                arr = np.array([float(v) for v in vals.split()])
            except ValueError:
                continue
            if key.strip().startswith('P_rect'):
                data[key.strip()] = arr.reshape(3, 4)
    return data


def main():
    parser = argparse.ArgumentParser(description='KITTI calib_cam_to_cam.txt → 项目标定 YAML')
    parser.add_argument('calib', help='calib_cam_to_cam.txt 路径')
    parser.add_argument('--width', type=int, required=True, help='图像宽 (raw:1392, 检测:1242)')
    parser.add_argument('--height', type=int, required=True, help='图像高 (raw:512, 检测:375)')
    parser.add_argument('--out', default='configs/kitti', help='输出目录 (默认: configs/kitti)')
    args = parser.parse_args()

    data = parse_kitti_calib(args.calib)
    if 'P_rect_02' not in data or 'P_rect_03' not in data:
        raise SystemExit('未找到 P_rect_02/P_rect_03，请确认是 KITTI calib_cam_to_cam.txt')

    K_l = data['P_rect_02'][:, :3]
    K_r = data['P_rect_03'][:, :3]
    fx = K_l[0, 0]
    # 基线：视差公式 d = fx*b/z，无穷远点视差 = (P02[0,3] - P03[0,3])/z
    # 注意 Scene Flow 的 P_rect_02[0,3] 可能不为 0，必须用差值
    b = (data['P_rect_02'][0, 3] - data['P_rect_03'][0, 3]) / fx  # 右相机相对左相机的 x 平移（米）
    print(f'左内参: fx={fx:.1f} fy={K_l[1, 1]:.1f} cx={K_l[0, 2]:.1f} cy={K_l[1, 2]:.1f}')
    print(f'基线 b = {b:.3f} m')
    if not 0.3 < b < 1.0:
        print('  ! 基线值异常(正常约0.54m)，请检查 calib 文件')

    eye3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    K_l = K_l.astype(float).tolist()
    K_r = K_r.astype(float).tolist()

    cam_template = {
        'model': 'kitti',
        'resolution': [args.width, args.height],
        'dist_coeffs': [0.0, 0.0, 0.0, 0.0, 0.0],  # KITTI 校正图已去畸变
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    left = dict(cam_template, camera_id='left', camera_matrix=K_l)
    right = dict(cam_template, camera_id='right', camera_matrix=K_r)
    with open(out / 'camera_left.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(left, f, default_flow_style=None, allow_unicode=True)
    with open(out / 'camera_right.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(right, f, default_flow_style=None, allow_unicode=True)

    stereo = {
        'baseline_mm': float(b * 1000),
        'left': {'camera_matrix': K_l, 'dist_coeffs': [0.0] * 5},
        'right': {'camera_matrix': K_r, 'dist_coeffs': [0.0] * 5},
        'stereo_extrinsics': {
            'rotation_matrix': eye3,
            'translation_vector': [float(b), 0.0, 0.0],
        },
    }
    with open(out / 'stereo_calibration.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(stereo, f, default_flow_style=None, allow_unicode=True)

    print(f'已生成: {out}/camera_left.yaml, camera_right.yaml, stereo_calibration.yaml')


if __name__ == '__main__':
    main()
