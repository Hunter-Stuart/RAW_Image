#!/usr/bin/env python
"""
场景自适应 ISP 验证实验

1. 判别准确率: 用 adaptive_params.classify_scene 对 AODRaw 各条件图像分类,
   与真实 tag 对照，输出混淆矩阵与准确率
2. 三方对比（按条件）: 自适应参数 vs 固定单套参数 vs 原图直测 的 mAP50
   - 自适应: 每图判别场景 → 套对应参数
   - 固定: 统一套用旧最优单套参数 (exposure 1.0 / contrast 1.4 / sharpen 1.5)
   - 原图: 不做任何处理
   模型: 训练后的 best_v2.pt

用法（本地）:
  python scripts/test_adaptive.py --n 30 --trained data/best_v2.pt
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
from isp_pipeline.adaptive_params import classify_scene, PARAMS

FIXED = dict(exposure_bias=1.0, contrast=1.4, sharpen_strength=1.5)  # 旧最优单套参数


def apply_params(img, p):
    gain = 2.0 ** p['exposure_bias']
    im = img.astype(np.float32) * gain
    if p['contrast'] != 1.0:
        gray = im.mean(axis=(0, 1), keepdims=True)
        im = (im - gray) * p['contrast'] + gray
    if p['sharpen_strength'] > 0:
        sigma = max(1.0, p['sharpen_strength'])
        ksize = int(sigma * 3) | 1
        im = im + min(p['sharpen_strength'] * 0.5, 2.0) * \
            (im - cv2.GaussianBlur(im, (ksize, ksize), sigma))
    return np.clip(im, 0, 255).astype(np.uint8)


def tag_to_class(tags):
    if 'low_light' in tags:
        return 'lowlight'
    if 'rain' in tags or 'fog' in tags:
        return 'weather'
    return 'normal'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=30, help='每条件图像数')
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

    conds = {'晴天': ('normal_light',), '暗光': ('low_light',), '雾天': ('fog',), '雨天': ('rain',)}
    img_dir = Path(args.img_dir)

    # ---------- 1. 判别准确率 ----------
    print('==== 1. 场景判别准确率 ====')
    cm = defaultdict(lambda: defaultdict(int))
    for cond, tags in conds.items():
        n_ok = n_total = 0
        for img in ann['images']:
            t = tuple(sorted(img.get('tag', [])))
            if not any(x in t for x in tags):
                continue
            if not any(a['category_id'] in class_map for a in gts_by_img[img['id']]):
                continue
            im = cv2.imread(str(img_dir / f"{Path(img['file_name']).stem}.JPG"))
            if im is None:
                continue
            pred = classify_scene(im)
            true = tag_to_class(t)
            cm[true][pred] += 1
            n_total += 1
            if pred == true:
                n_ok += 1
            if n_total >= args.n:
                break
        print(f'  {cond}: {n_ok}/{n_total} 正确 ({n_ok/max(n_total,1)*100:.0f}%)')
    print('  混淆矩阵(行=真实, 列=预测):')
    for t in ('lowlight', 'weather', 'normal'):
        print(f'    {t:>9}: ' + '  '.join(f'{p}:{cm[t][p]}' for p in ('lowlight', 'weather', 'normal')))

    # ---------- 2. 三方对比 ----------
    print('\n==== 2. 三方对比: 自适应 / 固定单套 / 原图 ====')
    results = {}
    for cond, tags in conds.items():
        dets = {'adaptive': defaultdict(list), 'fixed': defaultdict(list), 'raw': defaultdict(list)}
        gts_all = defaultdict(list)
        n = 0
        for img in ann['images']:
            t = tuple(sorted(img.get('tag', [])))
            if not any(x in t for x in tags):
                continue
            if not any(a['category_id'] in class_map for a in gts_by_img[img['id']]):
                continue
            im = cv2.imread(str(img_dir / f"{Path(img['file_name']).stem}.JPG"))
            if im is None:
                continue
            variants = {
                'adaptive': apply_params(im, PARAMS[classify_scene(im)]),
                'fixed': apply_params(im, FIXED),
                'raw': im,
            }
            for side, v in variants.items():
                r = model.predict(v, conf=0.05, verbose=False)[0]
                if r.boxes is not None and len(r.boxes):
                    for bb, sc, c in zip(r.boxes.xyxy.cpu().numpy(),
                                         r.boxes.conf.cpu().numpy(),
                                         r.boxes.cls.cpu().numpy().astype(int)):
                        dets[side][c].append((float(sc), *bb.tolist()))
            for a in gts_by_img[img['id']]:
                cid = class_map.get(a['category_id'])
                if cid is None:
                    continue
                bb = a['bbox']
                gts_all[cid].append((bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]))
            n += 1
            if n >= args.n:
                break
        ap = {}
        for side in dets:
            aps = [compute_ap50(dets[side].get(c, []), g) for c, g in gts_all.items() if g]
            ap[side] = float(np.mean(aps)) if aps else 0.0
        results[cond] = ap
        print(f'  {cond}: 自适应={ap["adaptive"]:.3f}  固定={ap["fixed"]:.3f}  原图={ap["raw"]:.3f}')

    out_dir = project_root / 'data/output/benchmark'
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_dir / 'adaptive_compare.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {out_dir}/adaptive_compare.json')


if __name__ == '__main__':
    main()
