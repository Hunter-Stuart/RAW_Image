#!/usr/bin/env python
"""
曝光补偿参数扫描（调优实验）

管线 exposure_bias 的语义是"对 Gamma 编码后的图像乘以 2^bias"，
因此在缓存的管线输出上直接做同样的增益即完全等价，无需重跑管线。
本脚本在 bias ∈ {0, 0.3, 0.5, 0.7, 1.0} 上扫描，输出各条件下
mAP50 与亮度，找到相对相机JPG的最优补偿值。

用法:
  python scripts/tune_exposure.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))

from aodraw_benchmark import (COND_GROUPS, build_class_map, compute_ap50)

BIASES = [0.7, 1.0, 1.3]
CONTRASTS = [1.0, 1.2, 1.4]
SHARPENS = [0.0, 1.5, 2.5]
WORK = project_root / 'data' / 'input' / 'aodraw_bench'
OUT = project_root / 'data' / 'output' / 'benchmark'


def main():
    ann = json.load(open(project_root.parent / 'AODRaw_annotations/AODRaw/annotations/test_annotations.json',
                         encoding='utf-8'))
    from ultralytics import YOLO
    model = YOLO(str(project_root / 'data/yolov8n.pt'))
    class_map = build_class_map(ann['categories'], model.names)

    gts_by_img = defaultdict(list)
    for a in ann['annotations']:
        gts_by_img[a['image_id']].append(a)

    # 选择与基准一致的前 N 张（有映射类别标注）
    selected = {}
    for cond, cfg in COND_GROUPS.items():
        picked = []
        for img in ann['images']:
            tags = tuple(sorted(img.get('tag', [])))
            if any(t in cfg['tags'] for t in tags):
                if any(a['category_id'] in class_map for a in gts_by_img[img['id']]):
                    picked.append(img)
                    if len(picked) >= 8:
                        break
        selected[cond] = picked

    results = {}
    print(f"{'bias/cont/sharp':>14} | " + ' | '.join(f'{c} mAP' for c in selected) + ' | 平均')
    for bias in BIASES:
        for contrast in CONTRASTS:
            for sharpen in SHARPENS:
                gain = 2.0 ** bias
                ap_sum = {}
                for cond, imgs in selected.items():
                    dets = defaultdict(list)
                    gts_all = defaultdict(list)
                    for img in imgs:
                        stem = Path(img['file_name']).stem
                        im = cv2.imread(str(WORK / 'A' / f'{stem}.png'))
                        im = im.astype(np.float32) * gain
                        if contrast != 1.0:  # 与管线 simple_tone_map 在 gamma 域等价
                            gray = im.mean(axis=(0, 1), keepdims=True)
                            im = (im - gray) * contrast + gray
                        if sharpen > 0:  # 与管线 unsharp 锐化等价
                            sigma = max(1.0, sharpen)
                            ksize = int(sigma * 3) | 1
                            blurred = cv2.GaussianBlur(im, (ksize, ksize), sigma)
                            im = im + min(sharpen * 0.5, 2.0) * (im - blurred)
                        im = np.clip(im, 0, 255).astype(np.uint8)
                        r = model.predict(im, conf=0.05, verbose=False)[0]
                        if r.boxes is not None and len(r.boxes):
                            for bb, s, c in zip(r.boxes.xyxy.cpu().numpy(),
                                               r.boxes.conf.cpu().numpy(),
                                               r.boxes.cls.cpu().numpy().astype(int)):
                                dets[c].append((float(s), *bb.tolist()))
                        ow = img.get('width', 6000); oh = img.get('height', 4000)
                        sx, sy = 1280.0 / ow, 720.0 / oh
                        for a in gts_by_img[img['id']]:
                            cid = class_map.get(a['category_id'])
                            if cid is None:
                                continue
                            bb = a['bbox']
                            gts_all[cid].append((bb[0] * sx, bb[1] * sy,
                                                 (bb[0] + bb[2]) * sx, (bb[1] + bb[3]) * sy))
                    aps = [compute_ap50(dets.get(c, []), g) for c, g in gts_all.items() if g]
                    ap_sum[cond] = float(np.mean(aps)) if aps else 0.0
                results[(bias, contrast, sharpen)] = ap_sum
                line = f"{bias}/{contrast}/{sharpen:>8.1f} | " + ' | '.join(f'{ap_sum[c]:.3f}' for c in selected) + \
                       f' | {np.mean(list(ap_sum.values())):.3f}'
                print(line)

    # 找最优组合（按各条件平均 mAP50）
    best = max(results, key=lambda k: np.mean(list(results[k].values())))
    print(f'\n最优 bias={best[0]} contrast={best[1]} sharpen={best[2]}'
          f'（各条件平均 mAP50 = {np.mean(list(results[best].values())):.3f}）')
    json.dump({f'{k[0]}/{k[1]}/{k[2]}': v for k, v in results.items()},
              open(OUT / 'tune_exposure.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
