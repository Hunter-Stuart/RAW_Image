"""
相机标定模块
使用棋盘格或 AprilTag 进行相机内参和畸变标定
支持单目标定（单目相机）和双目标定（双目相机）
"""

import numpy as np
import cv2
import yaml
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any


class ChessboardDetector:
    """棋盘格角点检测器"""

    def __init__(self, pattern_size: Tuple[int, int] = (9, 6),
                 square_size_mm: float = 25.0):
        """
        参数:
            pattern_size: 内角点数量 (宽, 高) —— 棋盘格有 9x6 个内角点
            square_size_mm: 每个棋盘格方块的边长 (mm)
        """
        self.pattern_size = pattern_size          # (columns, rows) of inner corners
        self.square_size = square_size_mm
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # 世界坐标系中的棋盘格角点 (z=0 平面)
        self.object_points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self.object_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        self.object_points *= square_size_mm

    def find_corners(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        在单张图像中查找棋盘格角点
        参数:
            image: 灰度或彩色图像
        返回:
            (found, corners) — found=True 且 corners 为 Nx2 亚像素角点坐标
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        found, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)

        if found:
            # 亚像素精度优化
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            return True, corners
        return False, None

    def draw_corners(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """在图像上绘制检测到的角点"""
        img = image.copy()
        cv2.drawChessboardCorners(img, self.pattern_size, corners, True)
        return img


class MonoCalibrator:
    """单目相机标定器"""

    def __init__(self, pattern_size: Tuple[int, int] = (9, 6),
                 square_size_mm: float = 25.0):
        self.detector = ChessboardDetector(pattern_size, square_size_mm)
        self.object_points_list: List[np.ndarray] = []   # 世界坐标点
        self.image_points_list: List[np.ndarray] = []     # 图像坐标点
        self.image_size: Optional[Tuple[int, int]] = None
        self._calibrated = False

        # 标定结果
        self.camera_matrix: Optional[np.ndarray] = None   # 内参矩阵 K
        self.dist_coeffs: Optional[np.ndarray] = None     # 畸变系数
        self.rvecs: Optional[List[np.ndarray]] = None     # 旋转向量
        self.tvecs: Optional[List[np.ndarray]] = None     # 平移向量
        self.reprojection_error: float = 0.0              # 重投影误差

    def add_image(self, image: np.ndarray) -> bool:
        """
        添加一张标定图像
        返回: True 如果成功检测到棋盘格
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        if self.image_size is None:
            self.image_size = (w, h)
        else:
            assert self.image_size == (w, h), "所有标定图像必须具有相同的分辨率"

        found, corners = self.detector.find_corners(gray)
        if found:
            self.object_points_list.append(self.detector.object_points.copy())
            self.image_points_list.append(corners)
            return True
        return False

    def calibrate(self) -> bool:
        """
        执行标定计算
        需要至少 10-20 张不同位姿的标定图像
        """
        if len(self.object_points_list) < 4:
            print(f"错误: 至少需要 4 张图像进行标定，当前仅 {len(self.object_points_list)} 张")
            return False

        if self.image_size is None:
            print("错误: 没有有效的标定图像")
            return False

        print(f"标定中... 使用 {len(self.object_points_list)} 张图像")
        print(f"图像分辨率: {self.image_size}")

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points_list,
            self.image_points_list,
            self.image_size,
            None, None
        )

        self.camera_matrix = K
        self.dist_coeffs = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.reprojection_error = ret
        self._calibrated = True

        return True

    def save_yaml(self, path: str, camera_id: str = "camera"):
        """保存标定结果为 YAML 文件"""
        if not self._calibrated:
            raise RuntimeError("请先执行 calibrate()")

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        data = {
            "camera_id": camera_id,
            "model": "pinhole",
            "resolution": list(self.image_size),
            "camera_matrix": [
                [float(fx), 0.0, float(cx)],
                [0.0, float(fy), float(cy)],
                [0.0, 0.0, 1.0],
            ],
            "dist_coeffs": self.dist_coeffs.flatten().tolist(),
            "reprojection_error_px": float(self.reprojection_error),
            "calibration_images": len(self.object_points_list),
            "pattern_size": list(self.detector.pattern_size),
            "square_size_mm": self.detector.square_size,
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        print(f"标定结果已保存: {path}")
        print(f"  内参矩阵 K:\n{self.camera_matrix}")
        print(f"  畸变系数: {self.dist_coeffs.flatten().tolist()}")
        print(f"  重投影误差: {self.reprojection_error:.4f} px")

    def get_results(self) -> Dict[str, Any]:
        """获取标定结果字典"""
        if not self._calibrated:
            return {}
        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.flatten().tolist(),
            "reprojection_error": float(self.reprojection_error),
            "image_size": list(self.image_size),
        }


class StereoCalibrator:
    """双目相机标定器"""

    def __init__(self, pattern_size: Tuple[int, int] = (9, 6),
                 square_size_mm: float = 25.0):
        self.detector = ChessboardDetector(pattern_size, square_size_mm)
        self.left_calib = MonoCalibrator(pattern_size, square_size_mm)
        self.right_calib = MonoCalibrator(pattern_size, square_size_mm)

        # 双目专用数据（配对检测）
        self.object_points_list: List[np.ndarray] = []
        self.left_points_list: List[np.ndarray] = []
        self.right_points_list: List[np.ndarray] = []
        self.image_size: Optional[Tuple[int, int]] = None

        # 标定结果
        self.R: Optional[np.ndarray] = None
        self.T: Optional[np.ndarray] = None
        self.E: Optional[np.ndarray] = None
        self.F: Optional[np.ndarray] = None
        self.reprojection_error: float = 0.0

    def add_stereo_pair(self, left_img: np.ndarray, right_img: np.ndarray) -> bool:
        """
        添加一对双目标定图像
        要求左右图都能检测到棋盘格
        """
        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY) if len(left_img.shape) == 3 else left_img
        gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY) if len(right_img.shape) == 3 else right_img

        h, w = gray_left.shape
        if self.image_size is None:
            self.image_size = (w, h)

        found_left, corners_left = self.detector.find_corners(gray_left)
        found_right, corners_right = self.detector.find_corners(gray_right)

        if found_left and found_right:
            self.object_points_list.append(self.detector.object_points.copy())
            self.left_points_list.append(corners_left)
            self.right_points_list.append(corners_right)
            # 同时加入单目标定
            self.left_calib.object_points_list.append(self.detector.object_points.copy())
            self.left_calib.image_points_list.append(corners_left)
            self.right_calib.object_points_list.append(self.detector.object_points.copy())
            self.right_calib.image_points_list.append(corners_right)
            if self.left_calib.image_size is None:
                self.left_calib.image_size = (w, h)
                self.right_calib.image_size = (w, h)
            return True
        return False

    def calibrate(self) -> bool:
        """执行双目标定"""
        if len(self.object_points_list) < 4:
            print(f"至少需要 4 对图像，当前 {len(self.object_points_list)}")
            return False

        print(f"双目标定中... 使用 {len(self.object_points_list)} 对图像")

        # 先标定单目
        self.left_calib.calibrate()
        self.right_calib.calibrate()

        # 再标定双目外参
        flags = cv2.CALIB_FIX_INTRINSIC
        ret, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(
            self.object_points_list,
            self.left_points_list,
            self.right_points_list,
            self.left_calib.camera_matrix,
            self.left_calib.dist_coeffs,
            self.right_calib.camera_matrix,
            self.right_calib.dist_coeffs,
            self.image_size,
            None, None, None, None,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        )

        self.R = R
        self.T = T
        self.E = E
        self.F = F
        self.reprojection_error = ret
        return True

    def save_yaml(self, path: str, baseline_mm: Optional[float] = None):
        """保存双目标定结果"""
        if baseline_mm is None:
            baseline_mm = float(np.linalg.norm(self.T) * 1000)

        data = {
            "baseline_mm": baseline_mm,
            "reprojection_error": float(self.reprojection_error),
            "image_size": list(self.image_size),
            "left": {
                "camera_matrix": self.left_calib.camera_matrix.tolist(),
                "dist_coeffs": self.left_calib.dist_coeffs.flatten().tolist(),
            },
            "right": {
                "camera_matrix": self.right_calib.camera_matrix.tolist(),
                "dist_coeffs": self.right_calib.dist_coeffs.flatten().tolist(),
            },
            "stereo_extrinsics": {
                "rotation_matrix": self.R.tolist(),
                "translation_vector": self.T.flatten().tolist(),
            },
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        print(f"双目标定结果已保存: {path}")
        print(f"  基线长度: {baseline_mm:.1f} mm")
        print(f"  重投影误差: {self.reprojection_error:.4f} px")
