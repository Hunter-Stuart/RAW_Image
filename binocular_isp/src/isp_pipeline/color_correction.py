"""
色彩校正模块
功能：色彩空间转换（相机RGB → sRGB）、色调映射、Gamma校正
"""

import numpy as np
from typing import Optional, Tuple


class ColorCorrector:
    """色彩校正器：相机色彩空间 → sRGB"""

    # 标准 sRGB 色域矩阵（CIE 1931 D65）
    SRGB_TO_XYZ = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])

    XYZ_TO_SRGB = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ])

    def __init__(self, color_matrix: Optional[np.ndarray] = None):
        """
        参数:
            color_matrix: 3x3 相机RGB→XYZ 色彩校正矩阵
                          None 则使用单位矩阵（需要后续白平衡和色彩调整）
        """
        self.color_matrix = color_matrix

    def apply_color_matrix(self, rgb: np.ndarray, 
                           color_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        应用色彩校正矩阵: 相机RGB → 标准色彩空间
        参数:
            rgb: HxWx3 RGB 图像
            color_matrix: 3x3 色彩校正矩阵
        返回:
            校正后的 RGB 图像
        """
        matrix = color_matrix if color_matrix is not None else self.color_matrix
        if matrix is None:
            return rgb

        h, w = rgb.shape[:2]
        rgb_flat = rgb.reshape(-1, 3).T  # 3xN
        corrected = (matrix @ rgb_flat).T  # Nx3
        corrected = corrected.reshape(h, w, 3)
        return np.clip(corrected, 0, None)

    def convert_to_srgb(self, rgb: np.ndarray, 
                        color_matrix_cam_to_xyz: Optional[np.ndarray] = None) -> np.ndarray:
        """
        转换到 sRGB 色彩空间
        流程：相机RGB → XYZ → sRGB
        参数:
            rgb: 相机 RGB 图像
            color_matrix_cam_to_xyz: 相机→XYZ 矩阵
        返回:
            sRGB 图像
        """
        if color_matrix_cam_to_xyz is not None:
            # 相机RGB → XYZ
            h, w = rgb.shape[:2]
            rgb_flat = rgb.reshape(-1, 3).T
            xyz = (color_matrix_cam_to_xyz @ rgb_flat).T
            xyz = xyz.reshape(h, w, 3)
            xyz = np.clip(xyz, 0, None)
        else:
            xyz = rgb.copy()

        # XYZ → sRGB（线性）
        h, w = xyz.shape[:2]
        xyz_flat = xyz.reshape(-1, 3).T
        srgb_linear = (self.XYZ_TO_SRGB @ xyz_flat).T
        srgb_linear = srgb_linear.reshape(h, w, 3)

        # 截断到 [0, 1]
        srgb_linear = np.clip(srgb_linear, 0, 1)

        return srgb_linear

    @staticmethod
    def gamma_correction(srgb_linear: np.ndarray, gamma: float = 2.2) -> np.ndarray:
        """
        Gamma 校正（sRGB 标准）
        sRGB 标准 Gamma 近似为 2.2
        参数:
            srgb_linear: 线性 sRGB [0, 1]
            gamma: Gamma 值
        返回:
            Gamma 校正后的 sRGB
        """
        return np.clip(srgb_linear ** (1.0 / gamma), 0, 1).astype(np.float32)

    @staticmethod
    def gamma_correction_srgb_standard(srgb_linear: np.ndarray) -> np.ndarray:
        """
        sRGB 标准 Gamma 校正（分段函数）
        线性部分: <= 0.0031308
        幂律部分: > 0.0031308
        """
        result = np.zeros_like(srgb_linear)
        mask = srgb_linear <= 0.0031308
        result[mask] = 12.92 * srgb_linear[mask]
        result[~mask] = 1.055 * (srgb_linear[~mask] ** (1.0 / 2.4)) - 0.055
        return np.clip(result, 0, 1).astype(np.float32)

    @staticmethod
    def inverse_gamma(srgb: np.ndarray, gamma: float = 2.2) -> np.ndarray:
        """反 Gamma 校正：sRGB → 线性 sRGB"""
        return np.clip(srgb ** gamma, 0, 1)

    @staticmethod
    def filmic_curve(srgb_linear: np.ndarray,
                     exposure_bias: float = 0.0,
                     shoulder: float = 1.0) -> np.ndarray:
        """
        ACES 风格 filmic S 曲线（Narkowicz 近似）
        - 暗部提升、高光软滚降、中间调保持对比（晴天差距主因的对应解法）
        - shoulder 控制曲线强度: 0=线性(无S), 1=标准ACES近似
        参考: ACES Filmic Tone Mapping Curves (Krzysztof Narkowicz 近似公开实现)
        """
        x = np.clip(srgb_linear * (2.0 ** exposure_bias), 0, 1)
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        f = np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)
        return (1.0 - shoulder) * x + shoulder * f

    @staticmethod
    def simple_tone_map(rgb: np.ndarray,
                        exposure_bias: float = 0.0,
                        contrast: float = 1.0,
                        brightness: float = 0.0,
                        saturation: float = 1.0) -> np.ndarray:
        """
        简单色调映射
        参数:
            rgb: HxWx3 float32 [0, 1] 图像
            exposure_bias: 曝光补偿（stop），+1 为亮度翻倍
            contrast: 对比度
            brightness: 亮度偏移
            saturation: 饱和度
        返回:
            色调映射后的图像
        """
        result = rgb.copy()

        # 曝光
        if exposure_bias != 0:
            result *= (2.0 ** exposure_bias)

        # 对比度和亮度
        if contrast != 1.0 or brightness != 0.0:
            gray = np.mean(result, axis=2, keepdims=True)
            result = (result - gray) * contrast + gray + brightness

        # 饱和度
        if saturation != 1.0:
            gray = np.mean(result, axis=2, keepdims=True)
            result = gray + (result - gray) * saturation

        return np.clip(result, 0, 1).astype(np.float32)

    @staticmethod
    def auto_levels(rgb: np.ndarray, clip_percent: float = 0.5) -> np.ndarray:
        """
        自动色阶：拉伸动态范围到 [0, 1]
        """
        result = np.zeros_like(rgb)
        for c in range(3):
            ch = rgb[:, :, c]
            low = np.percentile(ch, clip_percent)
            high = np.percentile(ch, 100 - clip_percent)
            if high > low:
                result[:, :, c] = np.clip((ch - low) / (high - low), 0, 1)
            else:
                result[:, :, c] = ch
        return result

    def __call__(self, rgb: np.ndarray,
                 color_matrix: Optional[np.ndarray] = None,
                 do_gamma: bool = True,
                 gamma: float = 2.2,
                 do_tone_map: bool = True,
                 tone_map_params: Optional[dict] = None) -> np.ndarray:
        """
        一键色彩校正：色彩矩阵 → sRGB → Gamma → 色调映射
        """
        # 色彩矩阵：传入的 color_matrix 视为 camera RGB → XYZ（rawpy 的 rgb_xyz_matrix），
        # 走 camera→XYZ→sRGB 标准链路；构造时设置的 self.color_matrix 视为 camera→sRGB，直接应用
        if color_matrix is not None:
            srgb = self.convert_to_srgb(rgb, color_matrix_cam_to_xyz=color_matrix)
        elif self.color_matrix is not None:
            srgb = np.clip(self.apply_color_matrix(rgb, None), 0, 1)
        else:
            srgb = self.convert_to_srgb(rgb)

        # 确保范围 [0, 1]
        srgb = np.clip(srgb, 0, 1)

        # 色调映射方式: 'filmic' 在 Gamma 之前作用于线性域（S曲线与显示编码分离）
        params = dict(tone_map_params or {})
        method = params.pop('method', 'linear')
        if do_tone_map and method == 'filmic':
            srgb = self.filmic_curve(
                srgb,
                exposure_bias=params.get('exposure_bias', 0.0),
                shoulder=params.get('shoulder', 1.0),
            )
            params.pop('exposure_bias', None)
            params.pop('shoulder', None)

        # Gamma
        if do_gamma:
            srgb = self.gamma_correction_srgb_standard(srgb)

        # 色调映射（微调参数，filmic 模式下可选叠加）
        if do_tone_map:
            srgb = self.simple_tone_map(srgb, **params)

        if srgb.dtype in (np.float32, np.float64):
            srgb = (srgb * 65535).astype(np.uint16)
        return srgb

