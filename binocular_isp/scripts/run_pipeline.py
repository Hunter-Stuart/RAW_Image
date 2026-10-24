#!/usr/bin/env python
"""
E-SIP Binocular ISP Pipeline - 主运行脚本
处理双目 RAW 图像 → 输出高质量 sRGB 图像
"""

import argparse
import sys
import yaml
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from isp_pipeline.pipeline import ISPPipeline, create_default_pipeline
from binocular.sync_control import StereoConsistencyController, StereoConsistencyChecker
from binocular.rectify import StereoRectifier, StereoCalibration
from config.camera_params import ConfigLoader
from utils.metadata import MetadataManager, ImageInfo


def parse_args():
    parser = argparse.ArgumentParser(
        description="E-SIP Binocular ISP Pipeline - 双目ISP处理管线"
    )
    parser.add_argument('--left', type=str, required=True,
                        help='左相机 RAW 文件路径或目录')
    parser.add_argument('--right', type=str, required=True,
                        help='右相机 RAW 文件路径或目录')
    parser.add_argument('--output', type=str, default='data/output',
                        help='输出目录 (默认: data/output)')
    parser.add_argument('--config', type=str, default='pipeline_config.yaml',
                        help='管线配置文件（相对 configs/ 目录）')
    parser.add_argument('--cam-left', type=str, default='camera_left.yaml',
                        help='左相机参数文件（相对 configs/ 目录）')
    parser.add_argument('--cam-right', type=str, default='camera_right.yaml',
                        help='右相机参数文件（相对 configs/ 目录）')
    parser.add_argument('--rectify', action='store_true',
                        help='启用去畸变和极线校正')
    parser.add_argument('--check-consistency', action='store_true',
                        help='检查左右图像一致性')
    parser.add_argument('--visualize', action='store_true',
                        help='显示调试可视化')
    parser.add_argument('--raw-only', action='store_true',
                        help='仅处理RAW图像，不调用校正（用于测试）')

    return parser.parse_args()


def process_single_pair(left_raw: str, right_raw: str,
                        pipeline: ISPPipeline,
                        consistency_ctrl: StereoConsistencyController,
                        rectifier: StereoRectifier,
                        checker: StereoConsistencyChecker,
                        output_dir: Path,
                        args) -> dict:
    """处理单对双目图像"""
    
    print(f"\n处理: {Path(left_raw).name} | {Path(right_raw).name}")
    
    results = {
        'left_raw': left_raw,
        'right_raw': right_raw,
        'success': False,
    }

    # 1. 处理左图
    print("  [1/4] 处理左图 RAW → sRGB...")
    left_srgb, left_meta = pipeline.process_file(left_raw)
    
    # 2. 处理右图
    print("  [2/4] 处理右图 RAW → sRGB...")
    # 使用与左图相同的白平衡增益
    right_srgb, right_meta = pipeline.process_file(right_raw)
    
    # 3. 去畸变和极线校正
    if args.rectify and rectifier is not None:
        print("  [3/4] 执行极线校正...")
        left_rect, right_rect = rectifier.rectify(left_srgb, right_srgb)
        
        # 验证校正质量
        verify_result = rectifier.verify_rectification(left_rect, right_rect)
        print(f"    校正质量: 平均垂直误差={verify_result['mean_vertical_error_px']:.3f}px, "
              f"通过0.5px阈值={verify_result['pass_THRESHOLD_0_5px']}")
        results['rectification'] = verify_result
    else:
        left_rect, right_rect = left_srgb, right_srgb

    # 4. 一致性检查
    if args.check_consistency:
        print("  [4/4] 检查双目一致性...")
        consistency_results = checker.check(left_rect, right_rect)
        print(f"    亮度差异: {consistency_results['brightness_diff']:.4f} "
              f"({'OK' if consistency_results['brightness_pass'] else 'FAIL'})")
        print(f"    颜色差异: {consistency_results['color_diff']:.4f} "
              f"({'OK' if consistency_results['color_pass'] else 'FAIL'})")
        print(f"    总体: {'OK 通过' if consistency_results['overall_pass'] else 'FAIL 未通过'}")
        results['consistency'] = consistency_results

    # 保存结果
    stem = Path(left_raw).stem
    out_subdir = output_dir / stem
    out_subdir.mkdir(parents=True, exist_ok=True)

    cv2_imwrite(str(out_subdir / 'left.png'), left_rect)
    cv2_imwrite(str(out_subdir / 'right.png'), right_rect)
    
    print(f"  结果已保存到: {out_subdir}")
    
    # 保存元数据
    meta_mgr = MetadataManager()
    meta_mgr.add('left_image', ImageInfo(
        file_path=str(out_subdir / 'left.png'),
        width=left_rect.shape[1],
        height=left_rect.shape[0],
        bit_depth=8,
        color_space='sRGB',
        camera_id='left',
    ).to_dict())
    meta_mgr.add('right_image', ImageInfo(
        file_path=str(out_subdir / 'right.png'),
        width=right_rect.shape[1],
        height=right_rect.shape[0],
        bit_depth=8,
        color_space='sRGB',
        camera_id='right',
    ).to_dict())
    meta_mgr.add('pipeline_config', pipeline.get_info())
    if 'rectification' in results:
        meta_mgr.add('rectification', results['rectification'])
    meta_mgr.save(str(out_subdir / 'metadata.json'))

    results['success'] = True
    return results


def cv2_imwrite(path: str, img):
    """cv2.imwrite 的别名（OpenCV 使用 BGR 顺序，确保正确）"""
    import cv2
    cv2.imwrite(path, img)


def main():
    args = parse_args()

    # 加载配置
    config_loader = ConfigLoader(str(Path(__file__).resolve().parent.parent / 'configs'))
    
    # 初始化 ISP 管线
    print("初始化 ISP 管线...")
    pipeline_config = config_loader.load_pipeline_config(args.config)
    pipeline = ISPPipeline(pipeline_config)
    print(f"  管线配置: {pipeline.get_info()}")

    # 初始化双目控制器
    consistency_ctrl = StereoConsistencyController()

    # 初始化校正器
    rectifier = None
    if args.rectify:
        print("加载标定参数...")
        left_cam = config_loader.load_camera_config('left', args.cam_left)
        right_cam = config_loader.load_camera_config('right', args.cam_right)

        # 加载双目外参配置
        stereo_config = config_loader.load('stereo_calibration.yaml')
        stereo_extrinsics = stereo_config.get('stereo_extrinsics', {})

        calib = StereoCalibration(
            left_camera_matrix=config_loader.config_to_camera_matrix(left_cam),
            left_dist_coeffs=config_loader.config_to_dist_coeffs(left_cam),
            right_camera_matrix=config_loader.config_to_camera_matrix(right_cam),
            right_dist_coeffs=config_loader.config_to_dist_coeffs(right_cam),
            rotation_matrix=ConfigLoader.config_to_rotation(stereo_extrinsics),
            translation_vector=ConfigLoader.config_to_translation(stereo_extrinsics),
            image_size=(left_cam.get('resolution', [1280, 720])[0],
                       left_cam.get('resolution', [1280, 720])[1]),
        )
        print("读取的 T 向量:", calib.translation_vector)
        print("读取的 R 矩阵:\n", calib.rotation_matrix)
        rectifier = StereoRectifier(calib)
        if calib.validate():
            print("  标定参数已加载")
        else:
            print("  ! 标定参数不完整，将跳过校正")
            rectifier = None

    # 初始化一致性检查器
    checker = StereoConsistencyChecker() if args.check_consistency else None

    # 准备输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 处理文件
    left_path = Path(args.left)
    right_path = Path(args.right)

    results_list = []

    if left_path.is_file() and right_path.is_file():
        # 单对处理
        result = process_single_pair(
            str(left_path), str(right_path),
            pipeline, consistency_ctrl, rectifier, checker,
            output_dir, args
        )
        results_list.append(result)

    elif left_path.is_dir() and right_path.is_dir():
        # 批量处理：匹配文件名
        left_files = sorted(left_path.glob('*.ARW')) + sorted(left_path.glob('*.DNG')) + \
                     sorted(left_path.glob('*.raw')) + sorted(left_path.glob('*.RAW'))
        right_files = sorted(right_path.glob('*.ARW')) + sorted(right_path.glob('*.DNG')) + \
                      sorted(right_path.glob('*.raw')) + sorted(right_path.glob('*.RAW'))

        # 按文件名前缀匹配
        left_map = {f.stem: f for f in left_files}
        right_map = {f.stem: f for f in right_files}
        common_stems = sorted(set(left_map.keys()) & set(right_map.keys()))

        print(f"\n找到 {len(common_stems)} 对匹配的 RAW 文件")
        for stem in common_stems:
            result = process_single_pair(
                str(left_map[stem]), str(right_map[stem]),
                pipeline, consistency_ctrl, rectifier, checker,
                output_dir, args
            )
            results_list.append(result)
    else:
        print("错误: 左右路径必须同时是文件或同时是目录")
        sys.exit(1)

    # 汇总
    success_count = sum(1 for r in results_list if r['success'])
    print(f"\n{'='*50}")
    print(f"处理完成: {success_count}/{len(results_list)} 对成功")
    print(f"输出目录: {output_dir.resolve()}")


if __name__ == '__main__':
    main()
