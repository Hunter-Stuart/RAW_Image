"""
ISP Pipeline 编排器
协调 RAW 读取 → 预处理 → 去马赛克 → 色彩校正 → 后处理 的完整流程
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

from .raw_reader import RAWReader, read_raw_file
from .pre_process import PreProcessor
from .demosaic import Demosaicer
from .color_correction import ColorCorrector
from .denoise_sharpen import Denoiser, Sharpener


class ISPPipeline:
    """
    ISP 管线主控器
    将各个处理模块串联为完整的 RAW → sRGB 流程
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        参数:
            config: 管线配置字典（可以从YAML加载）
        """
        self.config = config or {}

        # 初始化各模块（使用配置或默认参数）
        isp_cfg = self.config.get('isp', {})

        # 预处理
        bl = isp_cfg.get('black_level', 512)
        wl = isp_cfg.get('white_level', 16383)
        self.preprocessor = PreProcessor(black_level=bl, white_level=wl)

        # 去马赛克
        demosaic_method = isp_cfg.get('demosaic_method', 'opencv')
        self.demosaicer = Demosaicer(method=demosaic_method)

        # 色彩校正
        self.color_corrector = ColorCorrector()

        # 去噪
        dn_cfg = isp_cfg.get('denoise', {})
        self.denoiser = Denoiser(
            method=dn_cfg.get('method', 'none'),
            strength=dn_cfg.get('strength', 1.0)
        )

        # 锐化
        sp_cfg = isp_cfg.get('sharpen', {})
        self.sharpener = Sharpener(
            method=sp_cfg.get('method', 'none'),
            strength=sp_cfg.get('strength', 1.0)
        )

        # 输出配置
        self.output_cfg = self.config.get('output', {})
        self.target_width = self.output_cfg.get('width', 1280)
        self.target_height = self.output_cfg.get('height', 720)
        self.output_bit = self.output_cfg.get('bit_depth', 8)  # 8 or 16

    def process(self, raw_data: np.ndarray,
                metadata: Optional[Dict] = None,
                bayer_pattern: str = 'RGGB',
                wb_gains: Optional[Tuple[float, float, float, float]] = None,
                color_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        执行完整 ISP 管线处理
        流程:
        RAW Bayer → 黑电平校正 → 白平衡 → 去马赛克 → 色彩校正 → Gamma → 色调映射 → 去噪/锐化 → 缩放到目标尺寸

        参数:
            raw_data: Bayer RAW 数据 (HxW)
            metadata: 相机元数据
            bayer_pattern: Bayer 模式
            wb_gains: 白平衡增益，None自动估计
            color_matrix: 3x3 色彩校正矩阵

        返回:
            sRGB 图像 (HxWx3), 根据 output_bit 为 uint8 或 uint16
        """
        # 1. 预处理（黑电平 + 白平衡）
        data = self.preprocessor(
            raw_data,
            metadata=metadata,
            wb_gains=wb_gains,
            bayer_pattern=bayer_pattern
        )

        # 2. 去马赛克
        rgb = self.demosaicer(data, bayer_pattern)

        # 3. 色彩校正 + Gamma
        srgb_cfg = self.config.get('isp', {}).get('srgb', {})
        srgb = self.color_corrector(
            rgb.astype(np.float32) / (self.preprocessor.white_level or 65535.0),
            color_matrix=color_matrix,
            do_gamma=srgb_cfg.get('do_gamma', True),
            gamma=srgb_cfg.get('gamma', 2.2),
            do_tone_map=srgb_cfg.get('do_tone_map', True),
            tone_map_params={
                'exposure_bias': srgb_cfg.get('exposure_bias', 0.0),
                'contrast': srgb_cfg.get('contrast', 1.0),
                'brightness': srgb_cfg.get('brightness', 0.0),
                'saturation': srgb_cfg.get('saturation', 1.0),
            }
        )

        # 4. 去噪
        srgb = self.denoiser(srgb)

        # 5. 锐化
        srgb = self.sharpener(srgb)

        # 6. 缩放到目标分辨率
        if (srgb.shape[1] != self.target_width or 
            srgb.shape[0] != self.target_height):
            srgb = cv2.resize(srgb, (self.target_width, self.target_height),
                              interpolation=cv2.INTER_AREA)

        # 7. 输出格式转换
        if self.output_bit == 8:
            if srgb.dtype == np.uint16:
                srgb = (srgb / 257).astype(np.uint8)
            elif srgb.dtype == np.float32:
                srgb = (srgb * 255).astype(np.uint8)
        elif self.output_bit == 16:
            if srgb.dtype == np.uint8:
                srgb = (srgb.astype(np.uint16) * 257)
            elif srgb.dtype == np.float32:
                srgb = (srgb * 65535).astype(np.uint16)

        return srgb

    def process_file(self, raw_path: str,
                     bayer_pattern: str = 'RGGB',
                     wb_gains: Optional[Tuple] = None) -> Tuple[np.ndarray, Dict]:
        """
        从文件读取 RAW 并处理
        参数:
            raw_path: RAW 文件路径
            bayer_pattern: Bayer 模式
            wb_gains: 白平衡增益
        返回:
            (sRGB 图像, 元数据)
        """
        raw_data, metadata = read_raw_file(raw_path, bayer_pattern)

        # 从元数据提取色彩矩阵。
        # 实测 rawpy 0.27 的 rgb_xyz_matrix 为 XYZ→camera 方向（DNG ColorMatrix 约定，
        # 与 camera_whitebalance 乘数互逆），取逆得到 camera RGB → XYZ
        color_matrix = None
        if 'rgb_xyz_matrix' in metadata:
            M = np.array(metadata['rgb_xyz_matrix'], dtype=np.float64)
            color_matrix = np.linalg.inv(M).astype(np.float32)

        # 优先使用相机白平衡乘数（rgb_xyz_matrix 按相机WB后的值定义），缺失时回退灰世界
        if wb_gains is None and 'camera_wb' in metadata:
            wb = np.asarray(metadata['camera_wb'], dtype=np.float64)
            if wb[1] > 0:
                wb_gains = tuple(wb / wb[1])

        result = self.process(
            raw_data, metadata=metadata,
            bayer_pattern=bayer_pattern,
            wb_gains=wb_gains,
            color_matrix=color_matrix
        )

        return result, metadata

    def process_bayer_data(self, bayer_data: np.ndarray,
                           white_level: int = 16383,
                           bayer_pattern: str = 'RGGB',
                           black_level: Optional[int] = None) -> np.ndarray:
        """
        处理外部提供的 Bayer 数据（非 rawpy 读取）
        用于直接从传感器接收 Bayer 数据的场景
        black_level: None 时沿用管线配置（uint8 数据应显式传 0）
        """
        # 设置正确的白电平
        if white_level != self.preprocessor.white_level:
            self.preprocessor.white_level = white_level

        # 模拟元数据
        if black_level is None:
            black_level = self.preprocessor.black_level
        metadata = {
            'white_level': white_level,
            'black_level': black_level,
        }

        return self.process(bayer_data, metadata=metadata, bayer_pattern=bayer_pattern)

    def get_info(self) -> Dict:
        """获取当前管线配置信息"""
        return {
            'black_level': self.preprocessor.black_level,
            'white_level': self.preprocessor.white_level,
            'demosaic_method': self.demosaicer.method,
            'denoise_method': self.denoiser.method,
            'sharpen_method': self.sharpener.method,
            'target_resolution': (self.target_width, self.target_height),
            'output_bit_depth': self.output_bit,
        }


def create_default_pipeline(config_path: Optional[str] = None) -> ISPPipeline:
    """
    创建默认 ISP 管线实例
    参数:
        config_path: YAML 配置文件路径
    返回:
        ISPPipeline 实例
    """
    config = None
    if config_path and Path(config_path).exists():
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

    return ISPPipeline(config)
