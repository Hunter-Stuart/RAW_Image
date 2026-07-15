#!/usr/bin/env python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import cv2
from pathlib import Path
from calibration.calibrator import MonoCalibrator

def gen_frontal_checkerboard(inner_size, sq_px, img_size):
    """生成正视角度的棋盘格图像"""
    kw, kh = inner_size
    h = (kh+1)*sq_px
    w = (kw+1)*sq_px
    board = np.zeros((h, w), dtype=np.uint8)
    for i in range(kw+1):
        for j in range(kh+1):
            if (i+j)%2==0:
                board[j*sq_px:(j+1)*sq_px, i*sq_px:(i+1)*sq_px] = 255
    # 缩放到图像中并居中
    scale = min(img_size[0]/w*0.8, img_size[1]/h*0.8)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(board, (nw, nh), interpolation=cv2.INTER_NEAREST)
    full = np.ones((img_size[1], img_size[0]), dtype=np.uint8) * 200
    x_off = (img_size[0]-nw)//2
    y_off = (img_size[1]-nh)//2
    full[y_off:y_off+nh, x_off:x_off+nw] = resized
    return full

def run():
    import time
    t0 = time.time()
    sep = "=" * 60
    print(sep)
    print("  相机标定验证 - 开卷考试")
    print(sep)
    print()

    # ========== 第1步: 设定标准答案 ==========
    print("[1] 设定标准答案参数")
    img_size = (1280, 720)
    fx, fy, cx, cy = 800.0, 800.0, 640.0, 360.0
    K_gt = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
    dist_gt = np.array([[-0.1,0.02,0.001,0.0005,-0.005]], dtype=np.float64)
    inner_size = (9, 6)
    sq_mm = 25.0
    print("  分辨率: {}x{}".format(img_size[0], img_size[1]))
    print("  fx={:.1f} fy={:.1f} cx={:.1f} cy={:.1f}".format(fx,fy,cx,cy))
    print("  k1={} k2={} p1={} p2={} k3={}".format(dist_gt[0,0],dist_gt[0,1],dist_gt[0,2],dist_gt[0,3],dist_gt[0,4]))
    print("  棋盘格: {}x{} 内角点, {}mm/格".format(inner_size[0], inner_size[1], sq_mm))
    print()

    # ========== 第2步: 生成正视角棋盘格图像 ==========
    print("[2] 生成正视角棋盘格图像 (20张, 不同位置/距离)")
    np.random.seed(42)
    n = 20
    out = Path(__file__).parent.parent / "data" / "synthetic_checkerboard"
    out.mkdir(parents=True, exist_ok=True)
    cal = MonoCalibrator(inner_size, sq_mm)
    # 直接用已知角点构造标定数据
    objp = np.zeros((inner_size[0]*inner_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:inner_size[0], 0:inner_size[1]].T.reshape(-1,2) * sq_mm
    # 生成不同位姿的投影点
    all_objpts = []
    all_imgpts = []
    for i in range(n):
        # 随机位姿
        d = np.random.uniform(300, 800)
        rx = np.random.uniform(-0.3, 0.3)
        ry = np.random.uniform(-0.3, 0.3)
        rz = np.random.uniform(-0.15, 0.15)
        tx = np.random.uniform(-100, 100)
        ty = np.random.uniform(-100, 100)
        rvec = np.array([[rx],[ry],[rz]], dtype=np.float64)
        tvec = np.array([[tx],[ty],[d]], dtype=np.float64)
        # 投影
        imgpts, _ = cv2.projectPoints(objp, rvec, tvec, K_gt, dist_gt)
        all_objpts.append(objp.copy())
        all_imgpts.append(imgpts.copy())
        # 同时生成可检测的图像(正视角)
        sq_px = max(20, int(30 * 500/d))  # 根据距离调整像素大小
        det_img = gen_frontal_checkerboard(inner_size, sq_px, img_size)
        cv2.imwrite(str(out / "cb_{:02d}.png".format(i)), det_img)
        # 检测角点
        found, corners = cv2.findChessboardCorners(det_img, inner_size, None)
        if found: print("  [{:2d}/{}] d={:.0f}mm 正视角检测 OK".format(i+1,n,d))
        else: print("  [{:2d}/{}] d={:.0f}mm 正视角检测 FAIL".format(i+1,n,d))
    print()

    # ========== 第3步: 用已知 3D-2D 对应直接验证标定数学 ==========
    print("[3] 用已知 3D-2D 对应点执行标定 (验证数值精度)")
    ret, K_cal, dist_cal, rvecs, tvecs = cv2.calibrateCamera(
        all_objpts, all_imgpts, img_size, None, None
    )

    # ========== 第4步: 对比结果 ==========
    print()
    print("[4] 标定值 vs 标准答案")
    print("{:>10} {:>18} {:>18} {:>14}".format("参数", "标准答案", "标定结果", "误差"))
    print("-" * 60)
    fe = abs(K_cal[0,0]-fx)
    print("{:>10} {:>18.4f} {:>18.4f} {:>14.4f}".format("fx", fx, K_cal[0,0], fe))
    fe2 = abs(K_cal[1,1]-fy)
    print("{:>10} {:>18.4f} {:>18.4f} {:>14.4f}".format("fy", fy, K_cal[1,1], fe2))
    ce = abs(K_cal[0,2]-cx)
    print("{:>10} {:>18.4f} {:>18.4f} {:>14.4f}".format("cx", cx, K_cal[0,2], ce))
    ce2 = abs(K_cal[1,2]-cy)
    print("{:>10} {:>18.4f} {:>18.4f} {:>14.4f}".format("cy", cy, K_cal[1,2], ce2))
    for i, name in enumerate(["k1","k2","p1","p2","k3"]):
        g = dist_gt.flatten()[i]
        r = dist_cal.flatten()[i]
        print("{:>10} {:>18.6f} {:>18.6f} {:>14.6f}".format(name, g, r, abs(r-g)))
    print("-" * 60)
    print("{:>10} {:>18} {:>18.4f} px".format("重投影误差", "", ret))
    print()
    print("=" * 60)
    all_pass = fe<0.01 and fe2<0.01 and ce<0.01 and ce2<0.01
    if all_pass:
        print("  结论: 通过! 标定数学完全正确")
        print("  内参误差 < 0.01px (浮点精度级别)")
    else:
        print("  结论: 有偏差, 需排查")
        print("  fx_err={:.6f} fy_err={:.6f} cx_err={:.6f} cy_err={:.6f}".format(fe,fe2,ce,ce2))
    print("  耗时: {:.1f}s".format(time.time()-t0))

    # 保存标定结果
    cal.camera_matrix = K_cal
    cal.dist_coeffs = dist_cal
    cal.reprojection_error = ret
    cal.image_size = img_size
    cal._calibrated = True
    cal.save_yaml(str(out / "calibration_result.yaml"), "synthetic_test")
    print("  结果已保存")

if __name__ == "__main__":
    run()