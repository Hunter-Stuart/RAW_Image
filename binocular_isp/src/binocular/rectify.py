"""
去畸变和极线校正模块
对双目图像进行去畸变和极线校正，使左右图同名点在同一水平线上
"""

import numpy as np
import cv2
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass


@dataclass
class StereoCalibration:
    """
    双目标定参数
    包含左右相机的内参、畸变参数和外参
    """
    # 左相机
    left_camera_matrix: np.ndarray = None       # 3x3 内参矩阵
    left_dist_coeffs: np.ndarray = None         # 畸变系数 (k1, k2, p1, p2, k3, ...)

    # 右相机
    right_camera_matrix: np.ndarray = None      # 3x3 内参矩阵
    right_dist_coeffs: np.ndarray = None        # 畸变系数

    # 外参（左→右）
    rotation_matrix: np.ndarray = None           # 3x3 旋转矩阵
    translation_vector: np.ndarray = None        # 平移向量 (3,)

    # 图像尺寸
    image_size: Tuple[int, int] = (1280, 720)   # (width, height)

    # 校正后的参数
    left_map_x: np.ndarray = None               # 左图校正映射 x
    left_map_y: np.ndarray = None               # 左图校正映射 y
    right_map_x: np.ndarray = None              # 右图校正映射 x
    right_map_y: np.ndarray = None              # 右图校正映射 y
    R1: np.ndarray = None                       # 左校正旋转
    R2: np.ndarray = None                       # 右校正旋转
    P1: np.ndarray = None                       # 左投影矩阵
    P2: np.ndarray = None                       # 右投影矩阵
    Q: np.ndarray = None                        # 视差→深度 矩阵
    roi_left: Tuple = None                      # 左图有效区域
    roi_right: Tuple = None                     # 右图有效区域

    def validate(self) -> bool:
        """验证标定参数是否完整"""
        return all([
            self.left_camera_matrix is not None,
            self.right_camera_matrix is not None,
            self.rotation_matrix is not None,
            self.translation_vector is not None,
        ])


class StereoRectifier:
    """
    双目图像校正器
    功能：
    1. 去畸变（去除镜头畸变）
    2. 极线校正（左右图行对齐）
    """

    def __init__(self, calibration: Optional[StereoCalibration] = None):
        """
        参数:
            calibration: 双目标定参数
        """
        self.calibration = calibration
        self._maps_computed = False

    def set_calibration(self, calibration: StereoCalibration):
        """设置标定参数"""
        self.calibration = calibration
        self._maps_computed = False

    def compute_rectification_maps(self, 
                                   alpha: float = 0.0,
                                   new_size: Optional[Tuple[int, int]] = None):
        """
        计算校正映射表（只需计算一次，后续直接 remap）
        参数:
            alpha: 裁剪参数，0=裁剪到最大有效区域，1=保留全部像素
            new_size: 校正后的图像尺寸 (width, height)
        """
        if self.calibration is None:
            raise ValueError("请先设置标定参数")

        if not self.calibration.validate():
            raise ValueError("标定参数不完整")

        h, w = self.calibration.image_size[1], self.calibration.image_size[0]
        if new_size is not None:
            w, h = new_size

        # OpenCV 立体校正
        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
            self.calibration.left_camera_matrix,
            self.calibration.left_dist_coeffs,
            self.calibration.right_camera_matrix,
            self.calibration.right_dist_coeffs,
            (w, h),
            self.calibration.rotation_matrix,
            self.calibration.translation_vector,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=alpha,
            newImageSize=(w, h),
        )

        # 计算映射表
        left_map1, left_map2 = cv2.initUndistortRectifyMap(
            self.calibration.left_camera_matrix,
            self.calibration.left_dist_coeffs,
            R1, P1, (w, h), cv2.CV_32FC1
        )
        right_map1, right_map2 = cv2.initUndistortRectifyMap(
            self.calibration.right_camera_matrix,
            self.calibration.right_dist_coeffs,
            R2, P2, (w, h), cv2.CV_32FC1
        )

        # 保存结果
        self.calibration.left_map_x = left_map1
        self.calibration.left_map_y = left_map2
        self.calibration.right_map_x = right_map1
        self.calibration.right_map_y = right_map2
        self.calibration.R1 = R1
        self.calibration.R2 = R2
        self.calibration.P1 = P1
        self.calibration.P2 = P2
        self.calibration.Q = Q
        self.calibration.roi_left = roi_left
        self.calibration.roi_right = roi_right

        self._maps_computed = True

    def rectify(self, left_img: np.ndarray, right_img: np.ndarray,
                interpolation: int = cv2.INTER_LINEAR) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行去畸变和极线校正
        参数:
            left_img: 左图 HxWxC
            right_img: 右图 HxWxC
            interpolation: 插值方法
        返回:
            (rectified_left, rectified_right)
        """
        if not self._maps_computed:
            self.compute_rectification_maps()

        cal = self.calibration

        left_rect = cv2.remap(left_img, cal.left_map_x, cal.left_map_y,
                               interpolation, borderMode=cv2.BORDER_CONSTANT)
        right_rect = cv2.remap(right_img, cal.right_map_x, cal.right_map_y,
                                interpolation, borderMode=cv2.BORDER_CONSTANT)

        return left_rect, right_rect

    def rectify_left(self, img: np.ndarray) -> np.ndarray:
        """仅校正左图"""
        if not self._maps_computed:
            self.compute_rectification_maps()
        return cv2.remap(img, self.calibration.left_map_x,
                         self.calibration.left_map_y,
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def rectify_right(self, img: np.ndarray) -> np.ndarray:
        """仅校正右图"""
        if not self._maps_computed:
            self.compute_rectification_maps()
        return cv2.remap(img, self.calibration.right_map_x,
                         self.calibration.right_map_y,
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def verify_rectification(self, left_rect: np.ndarray, 
                             right_rect: np.ndarray,
                             num_lines: int = 20) -> Dict[str, Any]:
        """
        验证极线校正效果
        在校正后的图像上画水平线检查同名点是否在同一行
        返回校正质量指标
        """
        h, w = left_rect.shape[:2]
        gray_left = cv2.cvtColor(left_rect, cv2.COLOR_RGB2GRAY) if len(left_rect.shape) == 3 else left_rect
        gray_right = cv2.cvtColor(right_rect, cv2.COLOR_RGB2GRAY) if len(right_rect.shape) == 3 else right_rect

        # 计算校正后的同名点垂直差异（均方根）
        # 使用 ORB 特征匹配采样
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(gray_left, None)
        kp2, des2 = orb.detectAndCompute(gray_right, None)

        vertical_errors = []
        if des1 is not None and des2 is not None and len(kp1) > 0 and len(kp2) > 0:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)[:100]

            for m in matches:
                pt1 = kp1[m.queryIdx].pt
                pt2 = kp2[m.trainIdx].pt
                vertical_errors.append(abs(pt1[1] - pt2[1]))

        result = {
            'num_matches': len(vertical_errors),
            'mean_vertical_error_px': float(np.mean(vertical_errors)) if vertical_errors else -1,
            'max_vertical_error_px': float(max(vertical_errors)) if vertical_errors else -1,
            'std_vertical_error_px': float(np.std(vertical_errors)) if vertical_errors else -1,
            'pass_THRESHOLD_0_5px': float(np.mean(vertical_errors)) < 0.5 if vertical_errors else False,
        }

        return result

    def create_calibration_from_files(self, 
                                       left_intrinsic: np.ndarray,
                                       left_dist: np.ndarray,
                                       right_intrinsic: np.ndarray,
                                       right_dist: np.ndarray,
                                       R: np.ndarray,
                                       T: np.ndarray,
                                       image_size: Tuple[int, int]) -> StereoCalibration:
        """从文件/参数创建标定数据结构"""
        self.calibration = StereoCalibration(
            left_camera_matrix=left_intrinsic,
            left_dist_coeffs=left_dist,
            right_camera_matrix=right_intrinsic,
            right_dist_coeffs=right_dist,
            rotation_matrix=R,
            translation_vector=T,
            image_size=image_size,
        )
        return self.calibration

    @staticmethod
    def draw_epipolar_lines(left_img: np.ndarray, 
                            right_img: np.ndarray,
                            num_lines: int = 20) -> np.ndarray:
        """
        绘制极线（用于可视化校正效果）
        返回左右拼接的图像，带水平极线
        """
        h, w = left_img.shape[:2]
        # 确保彩色
        if len(left_img.shape) == 2:
            left_color = cv2.cvtColor(left_img, cv2.COLOR_GRAY2RGB)
            right_color = cv2.cvtColor(right_img, cv2.COLOR_GRAY2RGB)
        else:
            left_color = left_img.copy()
            right_color = right_img.copy()

        # 拼接
        vis = np.hstack([left_color, right_color])

        # 画水平极线
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255),
                  (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        step = h // num_lines
        for i in range(0, h, step):
            color = colors[(i // step) % len(colors)]
            cv2.line(vis, (0, i), (2 * w, i), color, 1)

        return vis
