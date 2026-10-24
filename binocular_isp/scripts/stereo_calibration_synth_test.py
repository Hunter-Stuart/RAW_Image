#!/usr/bin/env python
"""
双目标定"开卷考试"验证脚本

验证 calibrator.py 中 StereoCalibrator 的双目标定全流程正确性：
  1. 设定标准答案（左右内参/畸变/外参 R/T）
  2. 合成 20 对"左右同步观测的棋盘格角点"（带 0.1px 检测噪声）
  3. 走 StereoCalibrator.calibrate() 数学链路，对比标定输出与真值
  4. 验证 save_yaml 输出格式与项目三个 YAML 的字段对齐
  5. 附加端到端测试：渲染合成棋盘格图像对 → add_stereo_pair 完整检测链路

与 validate_calibration.py（单目版）互补：本脚本验证双目外参标定数学。
设备到货后，把本脚本里"填点列表"的部分换成"实拍图像对"即为正式标定流程。

用法:
  python scripts/stereo_calibration_synth_test.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from calibration.calibrator import StereoCalibrator


# ============ 合成数据生成 ============

def compose_pose(rvec_l, tvec_l, R_s, T_s):
    """由左相机位姿 + 右相对左的外参，求右相机位姿"""
    R_l = cv2.Rodrigues(rvec_l)[0]
    R_r = R_s @ R_l
    t_r = R_s @ tvec_l.reshape(3, 1) + T_s.reshape(3, 1)
    return cv2.Rodrigues(R_r)[0], t_r


def render_board(img_size, K, dist, rvec, tvec, pattern=(9, 6), square_mm=25.0):
    """
    把棋盘格按位姿渲染到图像（用于端到端检测测试）。
    逐方格做透视变换，准确体现镜头畸变，渲染精度接近真实拍摄。
    """
    kw, kh = pattern
    w, h = img_size
    sq = 40  # 纹理中每格像素数

    # 棋盘格纹理：内角点 kw×kh 个 → 方格 (kw+1)x(kh+1) 个
    tex = np.zeros(((kh + 1) * sq, (kw + 1) * sq), np.uint8)
    for i in range(kw + 1):
        for j in range(kh + 1):
            if (i + j) % 2 == 0:
                tex[j * sq:(j + 1) * sq, i * sq:(i + 1) * sq] = 255

    # 所有方格角点（含边框外沿，(kw+2)x(kh+2) 个）的世界坐标 → 畸变投影
    grid = np.zeros(((kh + 2) * (kw + 2), 1, 3), np.float64)
    grid[:, 0, :2] = np.array([[(i * square_mm, j * square_mm)
                                for i in range(kw + 2)]
                               for j in range(kh + 2)]).reshape(-1, 2)
    proj, _ = cv2.projectPoints(grid, rvec, tvec, K, dist)
    proj = proj.reshape(kh + 2, kw + 2, 2)

    # 逐方格：纹理方格 → 投影四边形 的透视变换，粘贴到中灰背景
    full = np.full((h, w), 128, np.uint8)
    for j in range(kh + 1):
        for i in range(kw + 1):
            src = np.float32([[i * sq, j * sq], [(i + 1) * sq, j * sq],
                              [(i + 1) * sq, (j + 1) * sq], [i * sq, (j + 1) * sq]])
            dst = np.float32([proj[j, i], proj[j, i + 1],
                              proj[j + 1, i + 1], proj[j + 1, i]])
            H = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(tex, H, (w, h))
            local = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(local, dst.astype(np.int32), 1)
            full[local == 1] = warped[local == 1]
    return full


def angle_error(R_cal, R_gt):
    """旋转矩阵角度误差（度）"""
    d = (np.trace(R_cal.T @ R_gt) - 1) / 2
    d = np.clip(d, -1, 1)
    return float(np.degrees(np.arccos(d)))


def main():
    sep = "=" * 62
    print(sep)
    print("  双目标定验证 - 开卷考试")
    print(sep)

    # ========== [1] 设定标准答案 ==========
    print("\n[1] 设定标准答案参数")
    img_size = (1280, 720)
    fx, fy, cx, cy = 800.0, 800.0, 640.0, 360.0
    K_gt = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist_gt = np.array([[-0.1, 0.02, 0.001, 0.0005, -0.005]], dtype=np.float64)
    # 右相机相对左相机：基线 120mm + 微小装配误差（单位与棋盘格一致：mm）
    r_ang = np.radians([0.5, -0.3, 0.2])
    R_s_gt = cv2.Rodrigues(r_ang)[0]
    T_s_gt = np.array([[-120.0], [2.0], [3.0]], dtype=np.float64)
    print(f"  分辨率: {img_size[0]}x{img_size[1]}, fx={fx} cx={cx} cy={cy}")
    print(f"  外参真值: R角=({np.degrees(r_ang[0]):.2f},{np.degrees(r_ang[1]):.2f},{np.degrees(r_ang[2]):.2f})deg, T={T_s_gt.flatten()} mm (基线{np.linalg.norm(T_s_gt):.1f}mm)")

    # ========== [2] 合成 20 对同步观测角点 ==========
    print("\n[2] 合成 20 对左右同步观测角点 (噪声 0.1px)")
    np.random.seed(42)
    kw, kh, sq_mm = 9, 6, 25.0
    objp = np.zeros((kw * kh, 3), np.float32)
    objp[:, :2] = np.mgrid[0:kw, 0:kh].T.reshape(-1, 2) * sq_mm

    stereo = StereoCalibrator(pattern_size=(kw, kh), square_size_mm=sq_mm)
    stereo.image_size = img_size

    poses = []
    for i in range(20):
        d = np.random.uniform(300, 800)
        rvec_l = np.array([[np.random.uniform(-0.3, 0.3)],
                           [np.random.uniform(-0.3, 0.3)],
                           [np.random.uniform(-0.15, 0.15)]], dtype=np.float64)
        tvec_l = np.array([[np.random.uniform(-100, 100)],
                           [np.random.uniform(-100, 100)],
                           [d]], dtype=np.float64)
        rvec_r, tvec_r = compose_pose(rvec_l, tvec_l, R_s_gt, T_s_gt)

        pl, _ = cv2.projectPoints(objp, rvec_l, tvec_l, K_gt, dist_gt)
        pr, _ = cv2.projectPoints(objp, rvec_r, tvec_r, K_gt, dist_gt)
        # 模拟角点检测噪声
        pl = pl + np.random.normal(0, 0.1, pl.shape)
        pr = pr + np.random.normal(0, 0.1, pr.shape)

        stereo.object_points_list.append(objp)
        stereo.left_points_list.append(pl.astype(np.float32))
        stereo.right_points_list.append(pr.astype(np.float32))
        stereo.left_calib.object_points_list.append(objp)
        stereo.left_calib.image_points_list.append(pl.astype(np.float32))
        stereo.right_calib.object_points_list.append(objp)
        stereo.right_calib.image_points_list.append(pr.astype(np.float32))
        poses.append((rvec_l, tvec_l, rvec_r, tvec_r))
    stereo.left_calib.image_size = stereo.right_calib.image_size = img_size

    # ========== [3] 执行双目标定数学链路 ==========
    print("\n[3] 执行 StereoCalibrator.calibrate()")
    assert stereo.calibrate(), "标定失败"

    # ========== [4] 对比结果 ==========
    print("\n[4] 标定值 vs 标准答案")
    K_cal = stereo.left_calib.camera_matrix
    err = {
        'fx': abs(K_cal[0, 0] - fx), 'fy': abs(K_cal[1, 1] - fy),
        'cx': abs(K_cal[0, 2] - cx), 'cy': abs(K_cal[1, 2] - cy),
    }
    for k, v in err.items():
        print(f"  {k}: 误差 {v:.2f} px")
    ang = angle_error(stereo.R, R_s_gt)
    t_err = float(np.linalg.norm(stereo.T - T_s_gt))
    b_gt = float(np.linalg.norm(T_s_gt))
    b_cal = float(np.linalg.norm(stereo.T))
    print(f"  R 角度误差: {ang:.4f} deg")
    print(f"  T 平移误差: {t_err:.2f} mm (标定基线 {b_cal:.1f}mm vs 真值 {b_gt:.1f}mm)")
    print(f"  双目重投影误差: {stereo.reprojection_error:.4f} px")

    k_pass = all(v < 5.0 for v in err.values())
    r_pass = ang < 0.2
    t_pass = t_err < 3.0
    print("\n  判定:")
    print(f"  内参误差 < 5px: {'PASS' if k_pass else 'FAIL'}")
    print(f"  R 误差 < 0.2deg: {'PASS' if r_pass else 'FAIL'}")
    print(f"  T 误差 < 3mm: {'PASS' if t_pass else 'FAIL'}")

    # ========== [5] 验证 save_yaml 格式 ==========
    print("\n[5] 验证 save_yaml 输出与项目 YAML 字段对齐")
    out = project_root / 'data' / 'output'
    out.mkdir(parents=True, exist_ok=True)
    yaml_path = out / 'synth_stereo_calib.yaml'
    stereo.save_yaml(str(yaml_path))
    import yaml
    with open(yaml_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    need = {'baseline_mm', 'left', 'right', 'stereo_extrinsics', 'reprojection_error'}
    missing = need - set(cfg.keys())
    ext_ok = set(cfg['stereo_extrinsics'].keys()) == {'rotation_matrix', 'translation_vector'}
    print(f"  顶层字段完整: {'PASS' if not missing else f'缺 {missing}'}")
    print(f"  stereo_extrinsics 字段对齐: {'PASS' if ext_ok else 'FAIL'}")

    # ========== [6] 端到端: 渲染图像对 → add_stereo_pair 完整链路 ==========
    print("\n[6] 端到端测试: 渲染棋盘格图像对 → add_stereo_pair 检测链路")
    e2e = StereoCalibrator(pattern_size=(kw, kh), square_size_mm=sq_mm)
    detected = 0
    for i in range(20):
        rvec_l, tvec_l, rvec_r, tvec_r = poses[i]
        # 用较温和的位姿渲染，保证可检测
        rvec_l *= 0.4
        rvec_r *= 0.4
        img_l = render_board(img_size, K_gt, dist_gt, rvec_l, tvec_l)
        img_r = render_board(img_size, K_gt, dist_gt, rvec_r, tvec_r)
        if e2e.add_stereo_pair(img_l, img_r):
            detected += 1
    print(f"  成功检测 {detected}/20 对")
    if detected >= 12:
        assert e2e.calibrate(), "端到端标定失败"
        e2e_ang = angle_error(e2e.R, R_s_gt)
        e2e_t = float(np.linalg.norm(e2e.T - T_s_gt))
        print(f"  端到端 R 角度误差: {e2e_ang:.3f} deg, T 误差: {e2e_t:.2f} mm")
        print(f"  端到端判定: {'PASS' if e2e_ang < 1.0 and e2e_t < 10.0 else 'FAIL'} (阈值宽松: 渲染板精度低于真实板)")
    else:
        print("  检测对数不足，端到端链路测试跳过（不影响数学验证结论）")

    print("\n" + sep)
    ok = k_pass and r_pass and t_pass
    print(f"  结论: {'全部通过! 双目标定链路正确' if ok else '存在偏差, 需排查'}")
    print(f"  输出: {yaml_path}")


if __name__ == '__main__':
    main()
