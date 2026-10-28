#!/usr/bin/env python
"""
晴天评估学习：局部对比度假设验证（CLAHE 实验）

晴天差距主因假设: 色调曲线(filmic)与色彩适应(Bradford)均无效 →
剩余候选为"局部对比度"。本实验在晴天图像上对比:
  原图 / CLAHE 1.5 / CLAHE 2.0 / CLAHE 2.0 + 轻曝光
若 CLAHE 显著提升 mAP50 → 支持"局部对比度"假设，值得在管线
(仅 normal 场景组)实现 CLAHE 模块。

用法: python scripts/sunny_clahe_study.py --n 30 --trained data/best_v2.pt
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


def apply_clahe(img, clip, gain=1.0):
    im = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=30)
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

    sunny = [img for img in ann['images']
             if 'normal_light' in img.get('tag', [])
             and any(a['category_id'] in class_map for a in gts_by_img[img['id']])][:args.n]
    print(f'晴天样本: {len(sunny)} 张')

    img_dir = Path(args.img_dir)
    variants = {'raw': [], 'clahe15': [], 'clahe20': [], 'clahe20_exp': []}
    gts_all = defaultdict(list)
    for img in sunny:
        stem = Path(img['file_name']).stem
        im = cv2.imread(str(img_dir / f'{stem}.JPG'))
        if im is None:
            continue
        variants['raw'].append(im)
        variants['clahe15'].append(apply_clahe(im, 1.5))
        variants['clahe20'].append(apply_clahe(im, 2.0))
        variants['clahe20_exp'].append(apply_clahe(im, 2.0, gain=1.2))
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

    base = results['raw']
    print(f'\n结论: 原图={base:.3f} | CLAHE1.5={results["clahe15"]:.3f} '
          f'(Δ{results["clahe15"]-base:+.3f}) | CLAHE2.0={results["clahe20"]:.3f} '
          f'(Δ{results["clahe20"]-base:+.3f}) | CLAHE2.0+曝光={results["clahe20_exp"]:.3f} '
          f'(Δ{results["clahe20_exp"]-base:+.3f})')

    out_dir = project_root / 'data/output/benchmark'
    json.dump(results, open(out_dir / 'sunny_clahe_study.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    names = list(results)
    vals = [results[k] for k in names]
    colors = ['#d62728', '#1f77b4', '#1f77b4', '#2ca02c']
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.003, f'{v:.3f}', ha='center')
    ax.axhline(base, ls='--', c='red', lw=1)
    ax.set_ylabel('mAP50'); ax.set_title('晴天: 局部对比度(CLAHE)假设验证')
    fig.tight_layout()
    fig.savefig(out_dir / 'sunny_clahe_study.png', dpi=120)
    plt.close(fig)
    print(f'图表: {out_dir}/sunny_clahe_study.png')


if __name__ == '__main__':
    main()
