"""
配置文件模块
加载和验证 YAML 配置文件，提供统一的配置接口
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple


class ConfigLoader:
    """配置加载器"""

    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path.cwd() / 'configs'

    def load(self, config_name: str) -> Dict[str, Any]:
        """
        加载 YAML 配置文件
        参数:
            config_name: 文件名（如 'pipeline_config.yaml'）或完整路径
        返回:
            配置字典
        """
        config_path = Path(config_name)
        if not config_path.is_absolute():
            config_path = self.config_dir / config_path

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config or {}

    def load_pipeline_config(self, path: str = 'pipeline_config.yaml') -> Dict:
        """加载管线配置"""
        return self.load(path)

    def load_camera_config(self, side: str = 'left', path: Optional[str] = None) -> Dict:
        """加载相机配置"""
        if path is None:
            path = f'camera_{side}.yaml'
        return self.load(path)

    @staticmethod
    def merge_configs(base: Dict, override: Dict) -> Dict:
        """合并配置（override 覆盖 base）"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader.merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def config_to_camera_matrix(config: Dict, key: str = 'camera_matrix') -> np.ndarray:
        """从配置解析相机内参矩阵"""
        values = config.get(key)
        if values is None:
            return None
        return np.array(values, dtype=np.float64).reshape(3, 3)

    @staticmethod
    def config_to_dist_coeffs(config: Dict, key: str = 'dist_coeffs') -> np.ndarray:
        """从配置解析畸变系数"""
        values = config.get(key)
        if values is None:
            return np.zeros((5, 1), dtype=np.float64)
        return np.array(values, dtype=np.float64).reshape(-1, 1)

    @staticmethod
    def config_to_rotation(config: Dict, key: str = 'rotation_matrix') -> np.ndarray:
        """从配置解析旋转矩阵"""
        values = config.get(key)
        if values is None:
            return np.eye(3, dtype=np.float64)
        return np.array(values, dtype=np.float64).reshape(3, 3)

    @staticmethod
    def config_to_translation(config: Dict, key: str = 'translation_vector') -> np.ndarray:
        """从配置解析平移向量"""
        values = config.get(key)
        if values is None:
            return np.zeros((3, 1), dtype=np.float64)
        return np.array(values, dtype=np.float64).reshape(3, 1)


def create_default_pipeline_config() -> Dict:
    """创建默认管线配置"""
    return {
        'isp': {
            'black_level': 512,
            'white_level': 16383,
            'bayer_pattern': 'RGGB',
            'demosaic_method': 'opencv',
            'srgb': {
                'do_gamma': True,
                'gamma': 2.2,
                'do_tone_map': True,
                'exposure_bias': 0.0,
                'contrast': 1.0,
                'brightness': 0.0,
                'saturation': 1.0,
            },
            'denoise': {
                'method': 'none',
                'strength': 1.0,
            },
            'sharpen': {
                'method': 'none',
                'strength': 0.0,
            },
        },
        'output': {
            'width': 1280,
            'height': 720,
            'bit_depth': 8,
            'format': 'png',
            'color_space': 'sRGB',
        },
        'stereo': {
            'enabled': True,
            'rectify': True,
            'alpha': 0.0,
            'consistency_check': True,
            'wb_strategy': 'average',
        },
    }


def create_default_camera_config(camera_id: str = 'left') -> Dict:
    """创建默认相机配置（需要填入实际标定值）"""
    fx, fy = 600.0, 600.0  # 焦距（pixel）
    cx, cy = 640.0, 360.0  # 主点

    return {
        'camera_id': camera_id,
        'model': 'placeholder',
        'resolution': [1280, 720],
        'camera_matrix': [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        'dist_coeffs': [0.0, 0.0, 0.0, 0.0, 0.0],
        'rotation_matrix': [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        'translation_vector': [0.0, 0.0, 0.0],
    }


def create_default_stereo_config(baseline_mm: float = 120.0) -> Dict:
    """创建默认双目配置"""
    return {
        'baseline_mm': baseline_mm,
        'left': create_default_camera_config('left'),
        'right': create_default_camera_config('right'),
        # 右图相对左图的外参
        'stereo_extrinsics': {
            'rotation_matrix': [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            'translation_vector': [baseline_mm / 1000.0, 0.0, 0.0],
        },
    }
