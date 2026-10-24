"""
场景自适应 ISP 参数模块（任务驱动 ISP 的最小可行实现）

基于本项目实验数据确定场景判别规则与参数组：
  - dark_isp_study（40张暗光图×16组参数）: 暗光最优 = 提亮1档 + 不锐化 + 不加对比度
  - tune_exposure（27组参数×4条件）: 雨/雾最优 = 锐化增强 + 适度对比度
  - aodraw_benchmark 实测统计: 暗光亮度均值 49~59，雨/雾对比度 29~43，晴天对比度 48~64
     → 阈值: 亮度<70 判暗光；亮度正常但对比度<35 判雨/雾

用法:
    from isp_pipeline.adaptive_params import get_srgb_overrides
    params = get_srgb_overrides(img)   # {'exposure_bias':.., 'contrast':.., 'sharpen_strength':..}
"""

import cv2
import numpy as np

# 各场景的管线参数组（对应 pipeline_config.yaml 的 srgb 字段）
PARAMS = {
    'lowlight': dict(exposure_bias=1.0, contrast=1.0, sharpen_strength=0.0),   # 暗光: 只提亮
    'weather':  dict(exposure_bias=0.7, contrast=1.3, sharpen_strength=2.0),   # 雨/雾: 强锐化
    'normal':   dict(exposure_bias=0.3, contrast=1.2, sharpen_strength=1.0),   # 正常: 温和
}

BRIGHTNESS_THRESH = 70.0   # 灰度均值阈值（暗光判别）
CONTRAST_THRESH = 35.0     # 灰度标准差阈值（雨/雾判别）


def scene_stats(img):
    """计算场景统计量（img: BGR uint8 HxWx3）"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return {'brightness': float(gray.mean()), 'contrast': float(gray.std())}


def classify_scene(img, brightness_thresh=BRIGHTNESS_THRESH, contrast_thresh=CONTRAST_THRESH):
    """场景判别: lowlight / weather / normal"""
    s = scene_stats(img)
    if s['brightness'] < brightness_thresh:
        return 'lowlight'
    if s['contrast'] < contrast_thresh:
        return 'weather'
    return 'normal'


def get_srgb_overrides(img):
    """根据图像内容返回管线 srgb 参数覆盖（可直接喂给 simple_tone_map / unsharp）"""
    return dict(PARAMS[classify_scene(img)])
