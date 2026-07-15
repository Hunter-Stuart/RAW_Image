#!/usr/bin/env python
"""
批处理脚本：批量处理双目 RAW 图像
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from isp_pipeline.pipeline import ISPPipeline
from config.camera_params import ConfigLoader
import json


def parse_args():
    parser = argparse.ArgumentParser(description="批量处理双目 RAW 图像")
    parser.add_argument('--left-dir', type=str, required=True, help='左图目录')
    parser.add_argument('--right-dir', type=str, required=True, help='右图目录')
    parser.add_argument('--output', type=str, default='data/output', help='输出目录')
    parser.add_argument('--config', type=str, default='configs/pipeline_config.yaml')
    parser.add_argument('--extensions', type=str, default='.ARW,.DNG,.raw,.RAW',
                       help='RAW 文件扩展名（逗号分隔）')
    parser.add_argument('--no-rectify', action='store_true', help='跳过校正')
    return parser.parse_args()


def main():
    args = parse_args()
    config_loader = ConfigLoader(str(project_root / 'configs'))
    pipeline = ISPPipeline(config_loader.load_pipeline_config(args.config))

    exts = [e.strip() for e in args.extensions.split(',')]
    
    left_dir = Path(args.left_dir)
    right_dir = Path(args.right_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 按文件名匹配
    left_files = {}
    for ext in exts:
        for f in left_dir.glob(f'*{ext}'):
            left_files[f.stem] = f
    right_files = {}
    for ext in exts:
        for f in right_dir.glob(f'*{ext}'):
            right_files[f.stem] = f

    common = sorted(set(left_files.keys()) & set(right_files.keys()))
    print(f"找到 {len(common)} 对匹配文件")

    import cv2
    summary = []

    for i, stem in enumerate(common):
        left_path = left_files[stem]
        right_path = right_files[stem]
        print(f"[{i+1}/{len(common)}] {stem}")

        try:
            left_img, _ = pipeline.process_file(str(left_path))
            right_img, _ = pipeline.process_file(str(right_path))

            out_dir = output_dir / stem
            out_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(out_dir / 'left.png'), left_img)
            cv2.imwrite(str(out_dir / 'right.png'), right_img)

            summary.append({
                'stem': stem,
                'success': True,
                'left': str(out_dir / 'left.png'),
                'right': str(out_dir / 'right.png'),
            })
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            summary.append({'stem': stem, 'success': False, 'error': str(e)})

    # 生成汇总报告
    report_path = output_dir / 'batch_report.json'
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    success = sum(1 for s in summary if s['success'])
    print(f"\n完成: {success}/{len(summary)} 成功, 报告: {report_path}")


if __name__ == '__main__':
    main()
