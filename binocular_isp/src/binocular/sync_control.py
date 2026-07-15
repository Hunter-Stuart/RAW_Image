"""
双目一致性控制模块
确保左右相机的曝光、增益、白平衡等参数一致
保证左右图像在亮度和色彩上匹配，满足立体匹配需求
"""

import numpy as np
import cv2
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class CameraControlParams:
    """相机控制参数快照"""
    exposure_time: float = 0.0        # 曝光时间 (us)
    analog_gain: float = 1.0          # 模拟增益
    digital_gain: float = 1.0         # 数字增益
    white_balance: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)  # RGGB增益
    hdr_mode: bool = False            # HDR 模式
    aperture: float = 0.0             # 光圈

    def to_dict(self) -> Dict:
        return {
            'exposure_time': self.exposure_time,
            'analog_gain': self.analog_gain,
            'digital_gain': self.digital_gain,
            'white_balance_r': self.white_balance[0],
            'white_balance_g1': self.white_balance[1],
            'white_balance_g2': self.white_balance[2],
            'white_balance_b': self.white_balance[3],
            'hdr_mode': self.hdr_mode,
            'aperture': self.aperture,
        }


@dataclass
class BinocularFrame:
    """双目帧数据"""
    left_raw: Optional[np.ndarray] = None         # 左图 RAW Bayer
    right_raw: Optional[np.ndarray] = None        # 右图 RAW Bayer
    left_srgb: Optional[np.ndarray] = None        # 左图 sRGB
    right_srgb: Optional[np.ndarray] = None       # 右图 sRGB
    timestamp: float = 0.0                        # 时间戳 (s)
    frame_id: int = 0                             # 帧号
    metadata_left: Dict = field(default_factory=dict)
    metadata_right: Dict = field(default_factory=dict)
    control_params: Optional[CameraControlParams] = None


class StereoConsistencyController:
    """
    双目一致性控制器
    核心功能：
    1. 确保左右相机使用相同的曝光/增益/WB参数
    2. 使左右 ISP 处理参数一致
    3. 提供左右图像质量一致性检测
    """

    def __init__(self):
        self.shared_params: Optional[CameraControlParams] = None

    def determine_shared_params(self,
                                left_metadata: Dict,
                                right_metadata: Dict,
                                strategy: str = 'average') -> CameraControlParams:
        """
        根据左右相机元数据确定统一的控制参数
        策略:
            - 'average': 左右参数取平均
            - 'left': 以左相机为准
            - 'right': 以右相机为准
            - 'conservative': 取较保守（更亮、更高增益）的值
        """
        left_ctrl = self._extract_control_params(left_metadata)
        right_ctrl = self._extract_control_params(right_metadata)

        if strategy == 'left':
            self.shared_params = left_ctrl
        elif strategy == 'right':
            self.shared_params = right_ctrl
        elif strategy == 'average':
            self.shared_params = CameraControlParams(
                exposure_time=(left_ctrl.exposure_time + right_ctrl.exposure_time) / 2,
                analog_gain=(left_ctrl.analog_gain + right_ctrl.analog_gain) / 2,
                digital_gain=(left_ctrl.digital_gain + right_ctrl.digital_gain) / 2,
                white_balance=tuple(
                    (np.array(left_ctrl.white_balance) + np.array(right_ctrl.white_balance)) / 2
                ),
                hdr_mode=left_ctrl.hdr_mode or right_ctrl.hdr_mode,
            )
        elif strategy == 'conservative':
            gain_l = left_ctrl.analog_gain * left_ctrl.digital_gain * left_ctrl.exposure_time
            gain_r = right_ctrl.analog_gain * right_ctrl.digital_gain * right_ctrl.exposure_time
            if gain_l >= gain_r:
                self.shared_params = left_ctrl
            else:
                self.shared_params = right_ctrl

        return self.shared_params

    def get_shared_wb_gains(self) -> Optional[Tuple[float, float, float, float]]:
        """获取共享的白平衡增益"""
        if self.shared_params is None:
            return None
        return self.shared_params.white_balance

    def get_shared_isp_config(self, base_config: Dict) -> Dict:
        """
        根据共享控制参数生成统一的 ISP 配置
        """
        return base_config.copy()

    def _extract_control_params(self, metadata: Dict) -> CameraControlParams:
        """从相机元数据提取控制参数"""
        return CameraControlParams(
            exposure_time=metadata.get('exposure_time', 0.0),
            analog_gain=metadata.get('analog_gain', 1.0),
            digital_gain=metadata.get('digital_gain', 1.0),
            white_balance=tuple(metadata.get('white_balance', (1.0, 1.0, 1.0, 1.0))),
            hdr_mode=metadata.get('hdr_mode', False),
        )


class StereoConsistencyChecker:
    """双目一致性检测器：检查左右图质量是否一致"""

    def __init__(self,
                 max_brightness_diff: float = 0.1,
                 max_color_diff: float = 0.05,
                 max_sharpness_diff: float = 0.2):
        self.max_brightness_diff = max_brightness_diff
        self.max_color_diff = max_color_diff
        self.max_sharpness_diff = max_sharpness_diff

    def check(self, left_img: np.ndarray, right_img: np.ndarray) -> Dict[str, Any]:
        """
        检查左右图像的一致性
        返回包含各项指标的字典
        """
        if left_img.dtype == np.uint8:
            l = left_img.astype(np.float32) / 255.0
            r = right_img.astype(np.float32) / 255.0
        elif left_img.dtype == np.uint16:
            l = left_img.astype(np.float32) / 65535.0
            r = right_img.astype(np.float32) / 65535.0
        else:
            l = left_img.astype(np.float32)
            r = right_img.astype(np.float32)

        results = {}

        # 1. 亮度一致性
        l_gray = cv2.cvtColor(l, cv2.COLOR_RGB2GRAY)
        r_gray = cv2.cvtColor(r, cv2.COLOR_RGB2GRAY)
        brightness_diff = abs(float(np.mean(l_gray) - np.mean(r_gray)))
        results['brightness_diff'] = brightness_diff
        results['brightness_left'] = float(np.mean(l_gray))
        results['brightness_right'] = float(np.mean(r_gray))
        results['brightness_pass'] = brightness_diff < self.max_brightness_diff

        # 2. 颜色一致性
        l_mean_color = np.mean(l, axis=(0, 1))
        r_mean_color = np.mean(r, axis=(0, 1))
        color_diff = float(np.mean(np.abs(l_mean_color - r_mean_color)))
        results['color_diff'] = color_diff
        results['color_left'] = l_mean_color.tolist()
        results['color_right'] = r_mean_color.tolist()
        results['color_pass'] = color_diff < self.max_color_diff

        # 3. 清晰度一致性
        l_laplacian = cv2.Laplacian(l_gray, cv2.CV_32F).var()
        r_laplacian = cv2.Laplacian(r_gray, cv2.CV_32F).var()
        sharpness_diff = abs(l_laplacian - r_laplacian) / (max(l_laplacian, r_laplacian) + 1e-6)
        results['sharpness_left'] = float(l_laplacian)
        results['sharpness_right'] = float(r_laplacian)
        results['sharpness_diff_ratio'] = float(sharpness_diff)
        results['sharpness_pass'] = sharpness_diff < self.max_sharpness_diff

        # 4. 直方图一致性
        l_hist = cv2.calcHist([l], [0], None, [64], [0, 1]).flatten()
        r_hist = cv2.calcHist([r], [0], None, [64], [0, 1]).flatten()
        l_hist = l_hist / (l_hist.sum() + 1e-6)
        r_hist = r_hist / (r_hist.sum() + 1e-6)
        hist_corr = float(np.correlate(l_hist, r_hist)[0])
        results['histogram_correlation'] = hist_corr

        # 总体判定
        results['overall_pass'] = (
            results['brightness_pass'] and
            results['color_pass'] and
            results['sharpness_pass']
        )

        return results
