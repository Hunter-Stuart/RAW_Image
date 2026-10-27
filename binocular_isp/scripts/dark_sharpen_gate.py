#!/usr/bin/env python
"""
暗光创新点落地实验：信噪比门控锐化 (Noise-Gated Sharpening)

动机: 暗光专项发现"全局锐化在暗光下全面起反作用（-0.04~-0.07 mAP50）"，
机理是锐化放大了噪声。解决方案: 按局部信噪比(SNR)门控锐化强度——
  高SNR区域(清晰边缘) → 正常锐化（恢复纹理细节）
  低SNR区域(平坦/噪声区) → 弱化或禁止锐化（不放大噪声）
实现: 局部结构能量(边缘²)与局部高频能量(噪声)之比估计SNR，sigmoid门控。

对比: 原图 / 全局锐化(2.0) / 门控锐化 三方在暗光40张上的 mAP50
用法: python scripts/dark_sharpen_gate.py --n 40 --trained data/best_v2.pt
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
torch.backends.cudnn.enabled = False

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))

from aodraw_benchmark import build_class_map, compute_ap50


def noise_gated_sharpen(img, amount=2.0, blur_sigma=1.0, k=4.0):
    """信噪比门控锐化 v2: 以梯度幅值中位数×k 为界——只锐化显著边缘
    （暗光下噪声梯度弱且密集，中位数阈值能有效滤除；边缘梯度强且稀疏，被保留）"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    thr = k * np.median(grad)
    gate = np.clip((grad - thr) / (thr + 1e-6), 0, 1)
    gate = cv2.GaussianBlur(gate, (9, 9), 0)   # 平滑过渡避免硬边伪影
    blurred = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    detail = img.astype(np.float32) - blurred
    out = img.astype(np.float32) + amount * detail * gate[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def global_sharpen(img, amount=2.0, blur_sigma=1.0):
    blurred = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    out = img.astype(np.float32) + amount * (img.astype(np.float32) - blurred)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=40)
    parser.add_argument('--ann', default=str(project_root.parent /
                                             'AODRaw/annotations/test_annotations_downsample_scale3_bbox_min_size32.json'))
    parser.add_argument('--img-dir', default=str(project_root.parent /
                                                 'AODRaw_downsampled_srgb/AODRaw/images_downsampled_srgb'))
    parser.add_argument('--trained', default=str(project_root / 'data/best_v2.pt'))
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
    variants = {'raw': [], 'global': [], 'gated': [], 'gain': [], 'gain_gated': []}
    gts_all = defaultdict(list)
    for img in dark:
        stem = Path(img['file_name']).stem
        im = cv2.imread(str(img_dir / f'{stem}.JPG'))
        if im is None:
            continue
        gain_im = np.clip(im.astype(np.float32) * 2.0, 0, 255).astype(np.uint8)
        variants['raw'].append(im)
        variants['global'].append(global_sharpen(im, amount=2.0))
        variants['gated'].append(noise_gated_sharpen(im, amount=2.0))
        variants['gain'].append(gain_im)
        variants['gain_gated'].append(noise_gated_sharpen(gain_im, amount=2.0))
        for a in gts_by_img[img['id']]:
            cid = class_map.get(a['category_id'])
            if cid is None:
                continue
            bb = a['bbox']
            gts_all[cid].append((bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]))

    results = {}
    for name, imgs in variants.items():
        dets = defaultdict(list)
        for im in imgs:
            r = model.predict(im, conf=0.05, verbose=False)[0]
            if r.boxes is not None and len(r.boxes):
                for bb, sc, c in zip(r.boxes.xyxy.cpu().numpy(),
                                     r.boxes.conf.cpu().numpy(),
                                     r.boxes.cls.cpu().numpy().astype(int)):
                    dets[c].append((float(sc), *bb.tolist()))
        aps = [compute_ap50(dets.get(c, []), g) for c, g in gts_all.items() if g]
        results[name] = float(np.mean(aps)) if aps else 0.0
        print(f'  {name}: mAP50 = {results[name]:.3f}')

    print(f'\n结论: 原图={results["raw"]:.3f} | 全局锐化={results["global"]:.3f} '
          f'(Δ{results["global"]-results["raw"]:+.3f}) | 门控锐化={results["gated"]:.3f} '
          f'(Δ{results["gated"]-results["raw"]:+.3f})')
    print(f'      提亮1档={results["gain"]:.3f} (Δ{results["gain"]-results["raw"]:+.3f}) | '
          f'提亮+门控锐化={results["gain_gated"]:.3f} (Δ{results["gain_gated"]-results["raw"]:+.3f})')

    out_dir = project_root / 'data/output/benchmark'
    json.dump(results, open(out_dir / 'dark_gate_sharpen.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(7, 5))
    names = ['原图(raw)', '全局锐化(global)', '门控锐化(gated)']
    vals = [results['raw'], results['global'], results['gated']]
    colors = ['#d62728', '#1f77b4', '#2ca02c']
    bars = ax.bar(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005, f'{v:.3f}', ha='center')
    ax.set_ylabel('mAP50')
    ax.set_title('暗光: 门控锐化 vs 全局锐化（信噪比门控抑制噪声放大）')
    fig.tight_layout()
    fig.savefig(out_dir / 'dark_gate_sharpen.png', dpi=120)
    plt.close(fig)
    print(f'图表: {out_dir}/dark_gate_sharpen.png')


if __name__ == '__main__':
    main()
