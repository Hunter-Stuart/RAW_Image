"""
ISP 配置管理模块
提供 ISP 参数的加载、保存和管理功能
"""

import yaml
import copy
from pathlib import Path
from typing import Dict, Any, Optional, List


class ISPConfig:
    """ISP 配置类"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()

    @staticmethod
    def _default_config() -> Dict:
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

    def get(self, key: str, default=None):
        """获取配置值，支持点号分隔的路径（如 'isp.srgb.gamma'）"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    def set(self, key: str, value):
        """设置配置值，支持点号分隔的路径"""
        keys = key.split('.')
        target = self.config
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value

    def save(self, path: str):
        """保存配置到 YAML 文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def load(cls, path: str) -> 'ISPConfig':
        """从 YAML 文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return cls(config)

    def copy(self) -> 'ISPConfig':
        """深拷贝配置"""
        return ISPConfig(copy.deepcopy(self.config))

    def __repr__(self) -> str:
        return yaml.dump(self.config, default_flow_style=False, allow_unicode=True)
