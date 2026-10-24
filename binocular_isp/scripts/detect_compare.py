#!/usr/bin/env python
"""
训练前后检测对比（按条件分组，重点看暗光）

用 AODRaw 降采样 sRGB 图像 + 官方 test 标注，对比:
  A: COCO 预训练 yolov8n.pt（训练前）
  B: 服务器训练得到的 best.pt（训练后）
按 晴天/暗光/雾天/雨天 分组统计 mAP50，并输出:
  - 控制台对比表
  - train_compare.png（分组柱状图）
  - dark_grid.png（暗光样例: 训练前 vs 训练后检测框并排）

用法:
  python scripts/detect_compare.py --per-cond 10 \
      --ann ../AODRaw/annotations/test_annotations_downsample_scale3_bbox_min_size32.json \
      --img-dir ../AODRaw_downsampled_srgb/AODRaw/images_downsampled_srgb \
      --pretrained data/yolov8n.pt --trained data/best.pt
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
torch.backends.cudnn.enabled = False  # 兼容服务器 torch2.4 与 cuDNN9.2 的错配（CUDNN_STATUS_NOT_INITIALIZED）

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))

from aodraw_benchmark import COND_GROUPS, build_class_map, compute_ap50


def run_detections(model, img, conf=0.05, device=None):
    r = model.predict(img, conf=conf, verbose=False, device=device)[0]
    out = defaultdict(list)
    if r.boxes is not None and len(r.boxes):
        for bb, s, c in zip(r.boxes.xyxy.cpu().numpy(),
                            r.boxes.conf.cpu().numpy(),
                            r.boxes.cls.cpu().numpy().astype(int)):
            out[c].append((float(s), *bb.tolist()))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-cond', type=int, default=10)
    parser.add_argument('--ann', default=str(project_root.parent /
                                             'AODRaw/annotations/test_annotations_downsample_scale3_bbox_min_size32.json'))
    parser.add_argument('--img-dir', default=str(project_root.parent /
                                                 'AODRaw_downsampled_srgb/AODRaw/images_downsampled_srgb'))
    parser.add_argument('--pretrained', default=str(project_root / 'data/yolov8n.pt'))
    parser.add_argument('--trained', default=str(project_root / 'data/best.pt'))
    parser.add_argument('--full-val', action='store_true',
                        help='用全部验证集图像按条件评测（不抽样，样本量最大，结果最可信）')
    parser.add_argument('--device', default=None, help='推理设备，如 0(GPU) 或 cpu（默认自动）')
    args = parser.parse_args()

    ann = json.load(open(args.ann, encoding='utf-8'))
    from ultralytics import YOLO
    model_a = YOLO(args.pretrained)   # 训练前
    model_b = YOLO(args.trained)      # 训练后
    class_map = build_class_map(ann['categories'], model_a.names)

    gts_by_img = defaultdict(list)
    for a in ann['annotations']:
        gts_by_img[a['image_id']].append(a)

    selected = {}
    if args.full_val:
        # 全量模式: 每个条件包含该 tag 的全部图像（一个图可属多个条件，如 暗光+雨）
        for cond, cfg in COND_GROUPS.items():
            imgs = [img for img in ann['images']
                    if any(t in tuple(sorted(img.get('tag', []))) for t in cfg['tags'])]
            selected[cond] = imgs
            print(f'{cond}: 全量 {len(imgs)} 张')
    else:
        for cond, cfg in COND_GROUPS.items():
            picked = []
            for img in ann['images']:
                tags = tuple(sorted(img.get('tag', [])))
                if any(t in cfg['tags'] for t in tags):
                    if any(a['category_id'] in class_map for a in gts_by_img[img['id']]):
                        picked.append(img)
                        if len(picked) >= args.per_cond:
                            break
            selected[cond] = picked
            print(f'{cond}: 选中 {len(picked)} 张')

    img_dir = Path(args.img_dir)
    results = {}
    for cond, imgs in selected.items():
        dets = {'A': defaultdict(list), 'B': defaultdict(list)}
        gts_all = defaultdict(list)
        for n, img in enumerate(imgs):
            stem = Path(img['file_name']).stem
            im = cv2.imread(str(img_dir / f'{stem}.JPG'))
            if im is None:
                continue
            for side, m in (('A', model_a), ('B', model_b)):
                d = run_detections(m, im, device=args.device)
                for c, v in d.items():
                    dets[side][c].extend(v)
            if args.full_val and (n + 1) % 200 == 0:
                print(f'  {cond} 进度: {n+1}/{len(imgs)}')
            ow = img.get('width', 2000); oh = img.get('height', 1333)
            for a in gts_by_img[img['id']]:
                cid = class_map.get(a['category_id'])
                if cid is None:
                    continue
                bb = a['bbox']
                gts_all[cid].append((bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]))
        ap = {}
        for side in ('A', 'B'):
            aps = [compute_ap50(dets[side].get(c, []), g)
                   for c, g in gts_all.items() if g]
            ap[side] = float(np.mean(aps)) if aps else 0.0
        results[cond] = ap

    print(f"\n{'条件':>6} | {'训练前 mAP50':>12} | {'训练后 mAP50':>12} | {'提升':>8}")
    print('-' * 48)
    for cond in results:
        a, b = results[cond]['A'], results[cond]['B']
        print(f'{cond:>6} | {a:>12.3f} | {b:>12.3f} | {b-a:>+8.3f}')

    # 图表
    out_dir = project_root / 'data/output/benchmark'
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_dir / 'train_compare.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    conds = list(results.keys())
    x = np.arange(len(conds))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, [results[c]['A'] for c in conds], w, label='训练前(COCO预训练)')
    ax.bar(x + w/2, [results[c]['B'] for c in conds], w, label='训练后(AODRaw微调)')
    ax.set_xticks(x); ax.set_xticklabels(conds)
    ax.set_ylabel('mAP50'); ax.set_title('训练前后检测对比（重点看暗光）')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'train_compare.png', dpi=120)
    plt.close(fig)

    # 暗光样例并排图
    dark_imgs = selected.get('暗光', [])[:2]
    rows = []
    for img in dark_imgs:
        stem = Path(img['file_name']).stem
        im = cv2.imread(str(img_dir / f'{stem}.JPG'))
        if im is None:
            continue
        ra = model_a.predict(im, conf=0.25, verbose=False)[0].plot()
        rb = model_b.predict(im, conf=0.25, verbose=False)[0].plot()
        rows.append(np.vstack([
            cv2.putText(ra.copy(), 'before training', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2),
            cv2.putText(rb.copy(), 'after training', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)]))
    if rows:
        cv2.imwrite(str(out_dir / 'dark_grid.png'), np.hstack(rows))
    print(f'\n图表已保存: {out_dir}/train_compare.png, dark_grid.png')


if __name__ == '__main__':
    main()
