"""
可选去噪和锐化模块
提供多种去噪和锐化策略，可根据需求开关
"""

import numpy as np
import cv2
from typing import Optional, Dict, Any


class Denoiser:
    """去噪器"""

    METHOD_NONE = 'none'
    METHOD_BILATERAL = 'bilateral'       # 双边滤波
    METHOD_NLM = 'nlm'                   # 非局部均值去噪
    METHOD_MEDIAN = 'median'             # 中值滤波
    METHOD_GAUSSIAN = 'gaussian'         # 高斯滤波

    def __init__(self, method: str = 'none', strength: float = 1.0):
        """
        参数:
            method: 去噪方法
            strength: 去噪强度 (0~10), 越大去噪越强
        """
        self.method = method
        self.strength = strength

    def denoise(self, rgb: np.ndarray) -> np.ndarray:
        """
        对 RGB 图像执行去噪
        参数:
            rgb: HxWx3 float32 [0, 1] 或 uint16 RGB
        返回:
            去噪后的图像
        """
        if self.method == self.METHOD_NONE:
            return rgb

        # 转为 float32 [0, 1] 处理
        if rgb.dtype == np.uint16:
            img = rgb.astype(np.float32) / 65535.0
        elif rgb.dtype == np.uint8:
            img = rgb.astype(np.float32) / 255.0
        else:
            img = rgb.copy()

        if self.method == self.METHOD_BILATERAL:
            sigma_color = max(10, self.strength * 30)
            sigma_space = max(10, self.strength * 10)
            img = cv2.bilateralFilter(img, -1, sigma_color, sigma_space)

        elif self.method == self.METHOD_NLM:
            h = max(1, self.strength * 3)
            img = cv2.fastNlMeansDenoisingColored(
                (img * 255).astype(np.uint8), None, h, h, 7, 21
            ).astype(np.float32) / 255.0

        elif self.method == self.METHOD_MEDIAN:
            ksize = max(3, int(self.strength * 2 + 1) | 1)  # 确保奇数
            img = cv2.medianBlur((img * 255).astype(np.uint8), ksize).astype(np.float32) / 255.0

        elif self.method == self.METHOD_GAUSSIAN:
            sigma = max(0.5, self.strength * 1.5)
            ksize = int(sigma * 3) | 1  # 确保奇数
            if ksize < 3:
                ksize = 3
            img = cv2.GaussianBlur(img, (ksize, ksize), sigma)

        # 转回原始类型
        if rgb.dtype == np.uint16:
            img = (img * 65535).astype(np.uint16)
        elif rgb.dtype == np.uint8:
            img = (img * 255).astype(np.uint8)

        return img

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        return self.denoise(rgb)


class Sharpener:
    """锐化器"""

    METHOD_NONE = 'none'
    METHOD_UNSHARP = 'unsharp'         # 非锐化掩模
    METHOD_LAPLACIAN = 'laplacian'     # 拉普拉斯锐化

    def __init__(self, method: str = 'none', strength: float = 1.0):
        """
        参数:
            method: 锐化方法
            strength: 锐化强度 (0~10)
        """
        self.method = method
        self.strength = strength

    def sharpen(self, rgb: np.ndarray) -> np.ndarray:
        """
        对 RGB 图像执行锐化
        参数:
            rgb: HxWx3 RGB 图像
        返回:
            锐化后的图像
        """
        if self.method == self.METHOD_NONE or self.strength <= 0:
            return rgb

        # 转为 float32 处理
        if rgb.dtype == np.uint16:
            img = rgb.astype(np.float32) / 65535.0
        elif rgb.dtype == np.uint8:
            img = rgb.astype(np.float32) / 255.0
        else:
            img = rgb.copy()

        if self.method == self.METHOD_UNSHARP:
            # 高斯模糊
            sigma = max(1.0, self.strength)
            ksize = int(sigma * 3) | 1
            blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
            # 原图 + 权重 * (原图 - 模糊)
            amount = min(self.strength * 0.5, 2.0)
            img = img + amount * (img - blurred)

        elif self.method == self.METHOD_LAPLACIAN:
            # 拉普拉斯锐化
            kernel = np.array([
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]
            ], dtype=np.float32)
            laplacian = cv2.filter2D(img, -1, kernel)
            amount = min(self.strength * 0.3, 1.0)
            img = img + amount * laplacian

        img = np.clip(img, 0, 1)

        # 转回原始类型
        if rgb.dtype == np.uint16:
            img = (img * 65535).astype(np.uint16)
        elif rgb.dtype == np.uint8:
            img = (img * 255).astype(np.uint8)

        return img

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        return self.sharpen(rgb)
