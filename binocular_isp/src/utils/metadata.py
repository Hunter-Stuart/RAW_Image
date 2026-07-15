"""
工具模块：元数据管理
"""

import json
import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class MetadataManager:
    """元数据管理器：记录图像采集和处理的全链路元数据"""

    def __init__(self):
        self.metadata: Dict[str, Any] = {}

    def add(self, key: str, value: Any):
        """添加元数据"""
        self.metadata[key] = value

    def add_batch(self, data: Dict):
        """批量添加"""
        self.metadata.update(data)

    def to_dict(self) -> Dict:
        """导出为字典"""
        return dict(self.metadata)

    def save(self, path: str):
        """保存元数据到 JSON 文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False, default=str)

    @classmethod
    def load(cls, path: str) -> "MetadataManager":
        """从 JSON 文件加载"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        mgr = cls()
        mgr.metadata = data
        return mgr

    @staticmethod
    def create_timestamp() -> str:
        """生成时间戳字符串"""
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


class ImageInfo:
    """图像信息记录"""

    def __init__(self, 
                 file_path: str = "",
                 width: int = 0,
                 height: int = 0,
                 bit_depth: int = 8,
                 color_space: str = "sRGB",
                 timestamp: float = 0.0,
                 frame_id: int = 0,
                 camera_id: str = ""):
        self.file_path = file_path
        self.width = width
        self.height = height
        self.bit_depth = bit_depth
        self.color_space = color_space
        self.timestamp = timestamp
        self.frame_id = frame_id
        self.camera_id = camera_id

    def to_dict(self) -> Dict:
        return {
            'file_path': self.file_path,
            'width': self.width,
            'height': self.height,
            'bit_depth': self.bit_depth,
            'color_space': self.color_space,
            'timestamp': self.timestamp,
            'frame_id': self.frame_id,
            'camera_id': self.camera_id,
        }
