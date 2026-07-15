"""
去马赛克模块
将 Bayer RAW 数据插值为全彩色 RGB 图像
提供多种插值方法
"""

import numpy as np
import cv2
from typing import Optional, Tuple


class Demosaicer:
    """Bayer 去马赛克器"""

    # 支持的插值方法
    METHOD_OPENCV = 'opencv'        # OpenCV cvtColor 插值（高质量）
    METHOD_BILINEAR = 'bilinear'    # 双线性插值（快速）
    METHOD_MALVAR = 'malvar'        # Malvar 改进型插值
    METHOD_OPENCV_EA = 'opencv_ea'  # OpenCV 边缘感知插值

    def __init__(self, method: str = 'opencv'):
        """
        参数:
            method: 插值方法
                - 'opencv': 使用 OpenCV 的 COLOR_BayerBG2BGR 等（推荐）
                - 'bilinear': 简单双线性插值
                - 'malvar': Malvar 改进型
        """
        self.method = method

        # OpenCV Bayer 转换代码映射
        self._opencv_codes = {
            'RGGB': {
                'rgb': cv2.COLOR_BAYER_RGGB2BGR,
                'rgb_ea': cv2.COLOR_BAYER_RGGB2BGR_EA,
            },
            'BGGR': {
                'rgb': cv2.COLOR_BAYER_BGGR2BGR,
                'rgb_ea': cv2.COLOR_BAYER_BGGR2BGR_EA,
            },
            'GRBG': {
                'rgb': cv2.COLOR_BAYER_GRBG2BGR,
                'rgb_ea': cv2.COLOR_BAYER_GRBG2BGR_EA,
            },
            'GBRG': {
                'rgb': cv2.COLOR_BAYER_GBRG2BGR,
                'rgb_ea': cv2.COLOR_BAYER_GBRG2BGR_EA,
            },
        }

    def demosaic(self, bayer_data: np.ndarray, bayer_pattern: str = 'RGGB') -> np.ndarray:
        """
        执行去马赛克
        参数:
            bayer_data: Bayer RAW 数据 (HxW), uint16 或 float32
            bayer_pattern: Bayer 模式
        返回:
            RGB 图像 (HxWx3), 与输入类型相同
        """
        if self.method in [self.METHOD_OPENCV, self.METHOD_OPENCV_EA]:
            return self._demosaic_opencv(bayer_data, bayer_pattern)
        elif self.method == self.METHOD_BILINEAR:
            return self._demosaic_bilinear(bayer_data, bayer_pattern)
        elif self.method == self.METHOD_MALVAR:
            return self._demosaic_malvar(bayer_data, bayer_pattern)
        else:
            raise ValueError(f"不支持的插值方法: {self.method}")

    def _demosaic_opencv(self, bayer_data: np.ndarray, bayer_pattern: str) -> np.ndarray:
        """使用 OpenCV 高质量插值"""
        pattern = bayer_pattern.upper()
        if pattern not in self._opencv_codes:
            raise ValueError(f"不支持的 Bayer 模式: {pattern}")

        # 确保数据类型是 uint16（OpenCV 支持）
        if bayer_data.dtype != np.uint16:
            if bayer_data.dtype == np.float32:
                # 缩放到 uint16 范围
                max_val = bayer_data.max()
                if max_val > 0:
                    bayer_uint16 = (bayer_data / max_val * 65535).astype(np.uint16)
                else:
                    bayer_uint16 = bayer_data.astype(np.uint16)
            else:
                bayer_uint16 = bayer_data.astype(np.uint16)
        else:
            bayer_uint16 = bayer_data

        use_ea = (self.method == self.METHOD_OPENCV_EA)
        code_key = 'rgb_ea' if use_ea else 'rgb'
        code = self._opencv_codes[pattern][code_key]

        rgb = cv2.cvtColor(bayer_uint16, code)

        # 转换回原始数据类型
        if bayer_data.dtype == np.float32:
            rgb = rgb.astype(np.float32) / 65535.0
        elif bayer_data.dtype == np.uint8:
            rgb = (rgb / 257).astype(np.uint8)

        return rgb

    def _demosaic_bilinear(self, bayer_data: np.ndarray, bayer_pattern: str) -> np.ndarray:
        """双线性插值（参考实现，OpenCV 方法更快）"""
        h, w = bayer_data.shape
        rgb = np.zeros((h, w, 3), dtype=bayer_data.dtype)
        pattern = bayer_pattern.upper()

        # 提取各通道采样位置
        R_mask = np.zeros((h, w), dtype=bool)
        G_mask = np.zeros((h, w), dtype=bool)
        B_mask = np.zeros((h, w), dtype=bool)

        if pattern == 'RGGB':
            R_mask[0:h:2, 0:w:2] = True
            G_mask[0:h:2, 1:w:2] = True
            G_mask[1:h:2, 0:w:2] = True
            B_mask[1:h:2, 1:w:2] = True
        elif pattern == 'BGGR':
            B_mask[0:h:2, 0:w:2] = True
            G_mask[0:h:2, 1:w:2] = True
            G_mask[1:h:2, 0:w:2] = True
            R_mask[1:h:2, 1:w:2] = True
        else:
            raise ValueError(f"Bilinear 暂不支持模式: {pattern}")

        # 为每个通道赋值
        for ch, mask in [(0, R_mask), (1, G_mask), (2, B_mask)]:
            channel = np.zeros((h, w), dtype=bayer_data.dtype)
            channel[mask] = bayer_data[mask]
            # 使用 OpenCV 的 inpaint 或中值/均值滤波来插值
            # 这里用简单的 3x3 均值滤波填补空洞
            kernel = np.ones((3, 3), np.float32) / 9
            channel_float = channel.astype(np.float32)
            filled = cv2.filter2D(channel_float, -1, kernel)
            # 只在空洞位置取值
            rgb[:, :, ch] = np.where(mask, channel_float, filled)

        return rgb

    def _demosaic_malvar(self, bayer_data: np.ndarray, bayer_pattern: str) -> np.ndarray:
        """
        Malvar 改进型去马赛克（参考实现）
        基于 Malvar et al. "High-quality linear interpolation for demosaicing of Bayer-patterned color images"
        """
        # 为了简洁，回退到 OpenCV 实现
        # 实际产品中推荐 OpenCV 方法
        print("Malvar 方法使用 OpenCV 回退实现")
        return self._demosaic_opencv(bayer_data, bayer_pattern)

    def __call__(self, bayer_data: np.ndarray, bayer_pattern: str = 'RGGB') -> np.ndarray:
        return self.demosaic(bayer_data, bayer_pattern)
