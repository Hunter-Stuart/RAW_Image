"""
ISP 管线测试
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
from isp_pipeline.pipeline import ISPPipeline
from isp_pipeline.raw_reader import simulate_bayer_raw
from isp_pipeline.pre_process import PreProcessor
from isp_pipeline.demosaic import Demosaicer
from isp_pipeline.color_correction import ColorCorrector


def test_preprocessor():
    """测试预处理模块"""
    print("\n测试: 预处理模块")

    raw = np.ones((10, 10), dtype=np.uint16) * 1000

    pp = PreProcessor(black_level=512, white_level=16383)
    corrected = pp.subtract_black_level(raw)

    assert corrected[0, 0] == 1000 - 512
    print(f"  黑电平校正: {raw[0,0]} -> {corrected[0,0]} PASS")

    # 白平衡
    gains = (2.0, 1.0, 1.0, 1.5)
    wb_result = pp.white_balance(corrected, gains)
    print(f"  白平衡测试 PASS")
    return True


def test_demosaic():
    """测试去马赛克模块"""
    print("\n测试: 去马赛克模块")

    bayer = np.zeros((10, 10), dtype=np.uint16)
    bayer[0:10:2, 0:10:2] = 1000   # R
    bayer[0:10:2, 1:10:2] = 2000   # G1
    bayer[1:10:2, 0:10:2] = 2000   # G2
    bayer[1:10:2, 1:10:2] = 3000   # B

    demosaicer = Demosaicer(method='opencv')
    rgb = demosaicer(bayer, 'RGGB')

    assert rgb.shape == (10, 10, 3), f"输出形状错误: {rgb.shape}"
    print(f"  去马赛克: {bayer.shape} -> {rgb.shape} PASS")
    return True


def test_pipeline_with_simulated_data():
    """使用模拟的 RAW 数据测试管线完整流程"""
    print("=" * 50)
    print("测试: 使用模拟 RAW 数据运行完整 ISP 管线")
    print("=" * 50)

    h, w = 100, 100
    rgb_test = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_test[:h//3, :, 0] = 200     # 红
    rgb_test[h//3:2*h//3, :, 1] = 200  # 绿
    rgb_test[2*h//3:, :, 2] = 200    # 蓝

    # 模拟 Bayer
    bayer = simulate_bayer_raw(rgb_test)
    print(f"  Bayer 形状: {bayer.shape}, dtype: {bayer.dtype}")
    print(f"  Bayer 值范围: [{bayer.min()}, {bayer.max()}]")

    # 创建管线 - 配置关闭 resize
    pipeline = ISPPipeline({
        'output': {'width': w, 'height': h, 'bit_depth': 8},
        'isp': {
            'black_level': 0,
            'white_level': 255,
            'srgb': {'do_gamma': True, 'gamma': 2.2, 'do_tone_map': False},
            'denoise': {'method': 'none'},
            'sharpen': {'method': 'none'},
        }
    })

    result = pipeline.process_bayer_data(bayer, white_level=255, bayer_pattern='RGGB')

    print(f"  输出形状: {result.shape}, dtype: {result.dtype}")
    print(f"  输出值范围: [{result.min()}, {result.max()}]")

    # 检查
    assert result.shape == (h, w, 3), f"输出形状错误: {result.shape}"
    assert result.dtype == np.uint8, f"输出类型错误: {result.dtype}"
    assert result.max() > 0, "输出全黑!"

    print("  PASS: 管线完整流程测试通过")
    return True


def test_color_correction():
    """测试色彩校正模块"""
    print("\n测试: 色彩校正模块")
    
    cc = ColorCorrector()
    
    # 输入 [0,1] 范围的 RGB
    rgb = np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.5, 0.5, 0.5]],
    ], dtype=np.float32)
    
    # 转 sRGB
    srgb = cc.convert_to_srgb(rgb)
    assert srgb.shape == rgb.shape
    assert srgb.min() >= -0.01 and srgb.max() <= 1.01, f"sRGB 范围异常: [{srgb.min()}, {srgb.max()}]"
    
    # Gamma
    gamma_img = cc.gamma_correction_srgb_standard(np.clip(srgb, 0, 1))
    assert gamma_img.shape == rgb.shape
    assert gamma_img.min() >= 0 and gamma_img.max() <= 1
    
    print(f"  色彩校正: {rgb.shape} -> gamma 后范围 [{gamma_img.min():.3f}, {gamma_img.max():.3f}] PASS")
    return True


if __name__ == '__main__':
    ok = True
    ok &= test_preprocessor()
    ok &= test_demosaic()
    ok &= test_color_correction()
    ok &= test_pipeline_with_simulated_data()
    print(f"\n{'='*50}")
    print(f"{'所有测试通过!' if ok else '有测试失败!'}")
