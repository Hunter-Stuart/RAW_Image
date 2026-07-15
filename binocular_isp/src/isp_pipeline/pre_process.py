"""
RAW 预处理模块
功能：黑电平校正、白平衡、归一化
"""

import numpy as np
from typing import Dict, Optional, Tuple


class PreProcessor:
    """RAW 预处理器：黑电平校正 + 白平衡"""

    def __init__(self, black_level: int = 512, white_level: int = 16383):
        """
        参数:
            black_level: 黑电平值（典型值：512 for 12-bit, 2048 for 14-bit, 512 for Sony ARW）
            white_level: 白电平/饱和值（典型值：16383 for 14-bit）
        """
        self.black_level = black_level
        self.white_level = white_level

    def set_from_metadata(self, metadata: Dict) -> None:
        """从相机元数据设置黑电平/白电平"""
        if 'black_level' in metadata:
            self.black_level = int(metadata['black_level'])
        if 'white_level' in metadata:
            self.white_level = int(metadata['white_level'])

    def subtract_black_level(self, raw_data: np.ndarray) -> np.ndarray:
        """
        黑电平校正：减去黑电平值，负值截断为0
        输入: uint16 RAW Bayer 数据
        输出: uint16 校正后数据
        """
        corrected = raw_data.astype(np.float32) - self.black_level
        corrected = np.clip(corrected, 0, None)
        return corrected.astype(np.uint16)

    def white_balance(self, raw_data: np.ndarray, 
                      gains: Tuple[float, float, float, float],
                      bayer_pattern: str = 'RGGB') -> np.ndarray:
        """
        白平衡：对 Bayer 各通道乘以对应增益
        参数:
            raw_data: Bayer RAW 数据 (HxW)
            gains: (R_gain, G1_gain, G2_gain, B_gain)
            bayer_pattern: Bayer 模式
        返回:
            白平衡后的数据
        """
        h, w = raw_data.shape
        result = raw_data.astype(np.float32)
        r_gain, g1_gain, g2_gain, b_gain = gains
        pattern = bayer_pattern.upper()

        if pattern == 'RGGB':
            result[0:h:2, 0:w:2] *= r_gain   # R
            result[0:h:2, 1:w:2] *= g1_gain  # G1
            result[1:h:2, 0:w:2] *= g2_gain  # G2
            result[1:h:2, 1:w:2] *= b_gain   # B
        elif pattern == 'BGGR':
            result[0:h:2, 0:w:2] *= b_gain   # B
            result[0:h:2, 1:w:2] *= g1_gain  # G1
            result[1:h:2, 0:w:2] *= g2_gain  # G2
            result[1:h:2, 1:w:2] *= r_gain   # R
        else:
            raise ValueError(f"不支持的 Bayer 模式: {pattern}")

        # 截断到 [0, white_level]
        result = np.clip(result, 0, self.white_level)
        return result.astype(np.uint16)

    def normalize(self, raw_data: np.ndarray) -> np.ndarray:
        """
        归一化到 [0, 1] float32
        """
        return raw_data.astype(np.float32) / self.white_level

    def denormalize(self, norm_data: np.ndarray, dtype: np.dtype = np.uint16) -> np.ndarray:
        """
        从 [0,1] 反归一化回原始整数范围
        """
        data = np.clip(norm_data * self.white_level, 0, self.white_level)
        return data.astype(dtype)

    def compute_wb_gains_from_gray_world(self, raw_data: np.ndarray,
                                          bayer_pattern: str = 'RGGB',
                                          black_level: Optional[int] = None) -> Tuple[float, float, float, float]:
        """
        使用灰世界假设自动计算白平衡增益
        假设场景平均色=R=G=B，各通道增益使均值相等
        返回: (R_gain, G1_gain, G2_gain, B_gain)
        """
        if black_level is not None:
            data = raw_data.astype(np.float32) - black_level
            data = np.clip(data, 0, None)
        else:
            data = raw_data.astype(np.float32)

        from .raw_reader import RAWReader
        reader = RAWReader.__new__(RAWReader)
        reader.bayer_pattern = bayer_pattern
        channels = reader.get_bayer_channels(data)

        r_mean = np.mean(channels['R'])
        g1_mean = np.mean(channels['G1'])
        g2_mean = np.mean(channels['G2'])
        b_mean = np.mean(channels['B'])

        g_mean = (g1_mean + g2_mean) / 2.0

        # 以绿色为基准
        eps = 1e-6
        r_gain = g_mean / (r_mean + eps)
        g1_gain = 1.0
        g2_gain = 1.0
        b_gain = g_mean / (b_mean + eps)

        return (r_gain, g1_gain, g2_gain, b_gain)

    def __call__(self, raw_data: np.ndarray, 
                 metadata: Optional[Dict] = None,
                 wb_gains: Optional[Tuple[float, float, float, float]] = None,
                 bayer_pattern: str = 'RGGB') -> np.ndarray:
        """
        一键预处理：黑电平校正 + 白平衡
        参数:
            raw_data: 原始 Bayer 数据
            metadata: 相机元数据（用于自动设置黑电平/白电平）
            wb_gains: 白平衡增益，None则自动估计
            bayer_pattern: Bayer 模式
        返回:
            预处理后的 RAW 数据
        """
        # 从元数据更新黑电平/白电平
        if metadata is not None:
            self.set_from_metadata(metadata)

        # 黑电平校正
        data = self.subtract_black_level(raw_data)

        # 白平衡
        if wb_gains is None:
            wb_gains = self.compute_wb_gains_from_gray_world(data, bayer_pattern)

        data = self.white_balance(data, wb_gains, bayer_pattern)

        return data
