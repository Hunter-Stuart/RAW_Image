#!/usr/bin/env python
"""
暗光专项实验：ISP 参数 × 检测精度 系统研究

在暗光测试图像上扫描"曝光增益 × 对比度 × 锐化"参数组合，
用训练后的检测模型评测 mAP50，回答：
  1. 暗光条件下哪些 ISP 参数组合最优？
  2. 相对"原图直接检测"，ISP 参数能带来多大提升？
  （这正是"任务驱动 ISP"在暗光场景的定量证据）

图像变换与管线 simple_tone_map/unsharp 在 gamma 域等价，
参数即管线配置（exposure_bias/contrast/sharpen_strength）的对应值。

用法（本地）:
  python scripts/dark_isp_study.py --n 40 --trained data/best_v2.pt
用法（服务器 GPU，先 scp 脚本与模型路径相应调整）:
  ... --device 0 --ann <服务器标注路径> --img-dir <服务器val图目录>
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
torch.backends.cudnn.enabled = False  # 兼容服务器 torch2.4+cuDNN9.2 错配

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))

from aodraw_benchmark import build_class_map, compute_ap50

GAINS = [1.0, 1.4, 2.0, 2.8]        # 曝光增益（≈ 0 / 0.5 / 1.0 / 1.5 stop）
CONTRASTS = [1.0, 1.3]
SHARPENS = [0.0, 2.0]


def apply_params(img, gain, contrast, sharpen):
    im = img.astype(np.float32) * gain
    if contrast != 1.0:
        gray = im.mean(axis=(0, 1), keepdims=True)
        im = (im - gray) * contrast + gray
    if sharpen > 0:
        sigma = max(1.0, sharpen)
        ksize = int(sigma * 3) | 1
        im = im + min(sharpen * 0.5, 2.0) * (im - cv2.GaussianBlur(im, (ksize, ksize), sigma))
    return np.clip(im, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=40, help='暗光图像数量')
    parser.add_argument('--ann', default=str(project_root.parent /
                                             'AODRaw/annotations/test_annotations_downsample_scale3_bbox_min_size32.json'))
    parser.add_argument('--img-dir', default=str(project_root.parent /
                                                 'AODRaw_downsampled_srgb/AODRaw/images_downsampled_srgb'))
    parser.add_argument('--trained', default=str(project_root / 'data/best_v2.pt'))
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    ann = json.load(open(args.ann, encoding='utf-8'))
    from ultralytics import YOLO
    model = YOLO(args.trained)
    class_map = build_class_map(ann['categories'], model.names)

    gts_by_img = defaultdict(list)
    for a in ann['annotations']:
        gts_by_img[a['image_id']].append(a)

    dark = [img for img in ann['images']
            if 'low_light' in img.get('tag', [])
            and any(a['category_id'] in class_map for a in gts_by_img[img['id']])][:args.n]
    print(f'暗光样本: {len(dark)} 张')

    img_dir = Path(args.img_dir)
    images = []
    gts_all = defaultdict(list)
    for img in dark:
        stem = Path(img['file_name']).stem
        im = cv2.imread(str(img_dir / f'{stem}.JPG'))
        if im is None:
            continue
        images.append((stem, im))
        for a in gts_by_img[img['id']]:
            cid = class_map.get(a['category_id'])
            if cid is None:
                continue
            bb = a['bbox']
            gts_all[cid].append((bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]))
    print(f'可用图像: {len(images)} 张, GT 类别数: {len(gts_all)}')

    combos = [(g, c, s) for g in GAINS for c in CONTRASTS for s in SHARPENS]
    results = {}
    for gi, (gain, contrast, sharpen) in enumerate(combos):
        dets = defaultdict(list)
        for stem, im in images:
            im_t = apply_params(im, gain, contrast, sharpen)
            r = model.predict(im_t, conf=0.05, verbose=False, device=args.device)[0]
            if r.boxes is not None and len(r.boxes):
                for bb, sc, c in zip(r.boxes.xyxy.cpu().numpy(),
                                     r.boxes.conf.cpu().numpy(),
                                     r.boxes.cls.cpu().numpy().astype(int)):
                    dets[c].append((float(sc), *bb.tolist()))
        aps = [compute_ap50(dets.get(c, []), g) for c, g in gts_all.items() if g]
        m = float(np.mean(aps)) if aps else 0.0
        results[f'{gain}/{contrast}/{sharpen}'] = m
        print(f'[{gi+1}/{len(combos)}] gain={gain} contrast={contrast} sharpen={sharpen} -> mAP50={m:.3f}')

    baseline = results['1.0/1.0/0.0']
    best_key = max(results, key=results.get)
    print(f'\n基线(原图直测): {baseline:.3f}')
    print(f'最优组合 {best_key}: {results[best_key]:.3f} (提升 {results[best_key]-baseline:+.3f})')

    out_dir = project_root / 'data/output/benchmark'
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_dir / 'dark_isp_study.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    keys = sorted(results, key=results.get)
    vals = [results[k] for k in keys]
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#2ca02c' if k == best_key else ('#d62728' if k == '1.0/1.0/0.0' else '#1f77b4') for k in keys]
    ax.barh(keys, vals, color=colors)
    ax.axvline(baseline, ls='--', c='red', lw=1, label=f'baseline {baseline:.3f}')
    ax.set_xlabel('mAP50'); ax.set_title('暗光条件: ISP 参数组合 × 检测精度 (gain/contrast/sharpen)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'dark_isp_study.png', dpi=120)
    plt.close(fig)
    print(f'图表已保存: {out_dir}/dark_isp_study.png')


if __name__ == '__main__':
    main()
