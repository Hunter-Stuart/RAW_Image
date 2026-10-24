"""
RAW 图像读取模块
支持读取多种 RAW 格式（.ARW, .DNG, .CR2, .NEF, .RAW 等）
基于 rawpy（LibRaw 封装）实现
"""

import numpy as np
import rawpy
import cv2
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


# 常见的 Bayer 模式排列
BAYER_PATTERNS = {
    'RGGB': 0,  # 标准 Bayer
    'BGGR': 1,
    'GRBG': 2,
    'GBRG': 3,
}


class RAWReader:
    """RAW 文件读取器"""

    def __init__(self, file_path: str, bayer_pattern: str = 'RGGB'):
        """
        参数:
            file_path: RAW 文件路径
            bayer_pattern: Bayer 模式 ('RGGB', 'BGGR', 'GRBG', 'GBRG')
        """
        self.file_path = Path(file_path)
        self.bayer_pattern = bayer_pattern
        self._raw_data = None
        self._raw_object = None

    def read(self) -> np.ndarray:
        """
        读取 RAW 数据，返回原始 Bayer 阵列
        返回: shape=(H, W) 的 uint16 数组
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"RAW 文件不存在: {self.file_path}")

        self._raw_object = rawpy.imread(str(self.file_path))
        raw_visible = self._raw_object.raw_image_visible
        self._raw_data = np.array(raw_visible, dtype=np.uint16)
        return self._raw_data

    def read_with_metadata(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        读取 RAW 数据并提取相机元数据
        返回: (raw_data, metadata_dict)
        """
        raw_data = self.read()
        metadata = self._extract_metadata()
        return raw_data, metadata

    def _extract_metadata(self) -> Dict[str, Any]:
        """提取相机元数据"""
        if self._raw_object is None:
            return {}

        meta = {}
        try:
            # 黑电平（rawpy 新版本可能返回 list 而非 ndarray，统一用 np.asarray 兼容）
            black_level = self._raw_object.black_level_per_channel
            if black_level is not None:
                bl = np.asarray(black_level)
                meta['black_level'] = int(np.mean(bl))
                meta['black_level_per_channel'] = bl.tolist()

            # 白电平（饱和值）
            white_level = self._raw_object.white_level
            if white_level is not None:
                meta['white_level'] = int(white_level)

            # 相机白平衡
            cam_wb = self._raw_object.camera_whitebalance
            if cam_wb is not None:
                meta['camera_wb'] = np.asarray(cam_wb).tolist()

            #  daylight 白平衡
            day_wb = self._raw_object.daylight_whitebalance
            if day_wb is not None:
                meta['daylight_wb'] = np.asarray(day_wb).tolist()

            # 色彩矩阵（rawpy 0.27 起提供 rgb_xyz_matrix: camera RGB → XYZ，4x3 末行全0）
            rgb_xyz = self._raw_object.rgb_xyz_matrix
            if rgb_xyz is not None:
                meta['rgb_xyz_matrix'] = np.asarray(rgb_xyz)[:3, :3].tolist()

            # 相机型号（rawpy 0.27 起无 camera_model 属性，用 getattr 兜底）
            meta['camera_model'] = str(getattr(self._raw_object, 'camera_model', '') or '')

            # RAW 图像尺寸
            meta['raw_size'] = (
                int(self._raw_object.raw_image_visible.shape[1]),
                int(self._raw_object.raw_image_visible.shape[0])
            )

            # Bayer 模式（libraw 的 color_desc 命名与 OpenCV 约定差在 3、4 字符顺序，
            # 如 'RGBG' ≡ 'RGGB'，统一转换为项目约定）
            color_desc = getattr(self._raw_object, 'color_desc', None)
            if color_desc is not None:
                desc = color_desc.decode('utf-8') if isinstance(color_desc, bytes) else str(color_desc)
                meta['bayer_pattern'] = normalize_bayer_pattern(desc)

            # 是否裁剪了感光区域
            sizes = getattr(self._raw_object, 'sizes', None)
            if sizes is not None:
                meta['raw_sizes'] = {
                    'raw_height': sizes.raw_height,
                    'raw_width': sizes.raw_width,
                    'height': sizes.height,
                    'width': sizes.width,
                    'top_margin': sizes.top_margin,
                    'left_margin': sizes.left_margin,
                }

        except Exception as e:
            print(f"提取元数据时出错: {e}")

        return meta

    def get_bayer_channels(self, raw_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        分离 Bayer 通道（RGGB）
        返回: {'R': 2D, 'G1': 2D, 'G2': 2D, 'B': 2D}
        """
        h, w = raw_data.shape
        pattern = self.bayer_pattern.upper()

        if pattern == 'RGGB':
            R = raw_data[0:h:2, 0:w:2]
            G1 = raw_data[0:h:2, 1:w:2]
            G2 = raw_data[1:h:2, 0:w:2]
            B = raw_data[1:h:2, 1:w:2]
        elif pattern == 'BGGR':
            B = raw_data[0:h:2, 0:w:2]
            G1 = raw_data[0:h:2, 1:w:2]
            G2 = raw_data[1:h:2, 0:w:2]
            R = raw_data[1:h:2, 1:w:2]
        elif pattern == 'GRBG':
            G1 = raw_data[0:h:2, 0:w:2]
            R = raw_data[0:h:2, 1:w:2]
            B = raw_data[1:h:2, 0:w:2]
            G2 = raw_data[1:h:2, 1:w:2]
        elif pattern == 'GBRG':
            G1 = raw_data[0:h:2, 0:w:2]
            B = raw_data[0:h:2, 1:w:2]
            R = raw_data[1:h:2, 0:w:2]
            G2 = raw_data[1:h:2, 1:w:2]
        else:
            raise ValueError(f"不支持的 Bayer 模式: {pattern}")

        return {'R': R, 'G1': G1, 'G2': G2, 'B': B}

    def close(self):
        """释放资源"""
        if self._raw_object is not None:
            self._raw_object.close()
            self._raw_object = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def normalize_bayer_pattern(desc: str) -> str:
    """
    将 libraw/rawpy 的 color_desc 命名转换为项目约定（OpenCV 风格）。
    libraw 'RGBG' ≡ 本项目 'RGGB'：两者第 3、4 字符顺序相反。
    """
    valid = ('RGGB', 'BGGR', 'GRBG', 'GBRG')
    desc = (desc or '').strip().upper()
    if desc in valid:
        return desc
    if len(desc) == 4:
        swapped = desc[0] + desc[1] + desc[3] + desc[2]
        if swapped in valid:
            return swapped
    return 'RGGB'


def read_raw_file(file_path: str, bayer_pattern: str = 'RGGB') -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    快捷函数：读取 RAW 文件并返回数据和元数据
    参数:
        file_path: RAW 文件路径
        bayer_pattern: Bayer 模式
    返回:
        (raw_data, metadata)
    """
    reader = RAWReader(file_path, bayer_pattern)
    return reader.read_with_metadata()


def simulate_bayer_raw(rgb_image: np.ndarray, pattern: str = 'RGGB') -> np.ndarray:
    """
    从 RGB 图像模拟生成 Bayer RAW 数据（用于没有真实 RAW 时的测试）
    参数:
        rgb_image: HxWx3 uint8/uint16 RGB 图像
        pattern: Bayer 模式
    返回:
        HxW 单通道 Bayer 阵列
    """
    h, w = rgb_image.shape[:2]
    if len(rgb_image.shape) == 2 or rgb_image.shape[2] == 1:
        # 灰度图，复制到 3 通道
        rgb = np.stack([rgb_image.squeeze()] * 3, axis=2)
    else:
        rgb = rgb_image[:, :, :3]

    # 确保偶数尺寸
    h_even = h if h % 2 == 0 else h - 1
    w_even = w if w % 2 == 0 else w - 1
    rgb = rgb[:h_even, :w_even]

    bayer = np.zeros((h_even, w_even), dtype=rgb.dtype)

    if pattern == 'RGGB':
        bayer[0:h_even:2, 0:w_even:2] = rgb[0:h_even:2, 0:w_even:2, 0]  # R
        bayer[0:h_even:2, 1:w_even:2] = rgb[0:h_even:2, 1:w_even:2, 1]  # G1
        bayer[1:h_even:2, 0:w_even:2] = rgb[1:h_even:2, 0:w_even:2, 1]  # G2
        bayer[1:h_even:2, 1:w_even:2] = rgb[1:h_even:2, 1:w_even:2, 2]  # B
    elif pattern == 'BGGR':
        bayer[0:h_even:2, 0:w_even:2] = rgb[0:h_even:2, 0:w_even:2, 2]  # B
        bayer[0:h_even:2, 1:w_even:2] = rgb[0:h_even:2, 1:w_even:2, 1]  # G1
        bayer[1:h_even:2, 0:w_even:2] = rgb[1:h_even:2, 0:w_even:2, 1]  # G2
        bayer[1:h_even:2, 1:w_even:2] = rgb[1:h_even:2, 1:w_even:2, 0]  # R
    else:
        raise ValueError(f"不支持的 Bayer 模式: {pattern}")

    return bayer
