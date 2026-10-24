#!/usr/bin/env python
"""
AODRaw COCO 标注 → YOLO 训练格式转换

只保留与 COCO 80 类同名（可映射）的类别（约 33/63 类），生成 ultralytics
训练所需的标准目录结构:
  <out>/images/train -> <img-dir> 的符号链接（不复制图像）
  <out>/labels/train/<stem>.txt    （每行: cls cx cy w h，归一化坐标）
  <out>/data.yaml

用法（在服务器上执行，需先激活装有 ultralytics 的环境——仅用于取类别名表）:
  python scripts/coco_to_yolo.py \
      --ann ../AODRaw_annotations/AODRaw/annotations/train_annotations_downsample_scale3_bbox_min_size32.json \
      --img-dir ../AODRaw_downsampled_srgb/AODRaw/images_downsampled_srgb \
      --out ~/datasets/aodraw --img-ext .JPG
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# COCO 80 类别名表（id: name），转换时按名称匹配
COCO_NAMES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush',
}

# AODRaw 类别名的拼写修正
NAME_OVERRIDES = {'bicyle': 'bicycle', 'phone': 'cell phone'}


def build_class_map(ann_categories):
    """AODRaw类别id -> YOLO(COCO)类别id，按名称匹配"""
    name_to_id = {v: k for k, v in COCO_NAMES.items()}
    mapping = {}
    for c in ann_categories:
        name = NAME_OVERRIDES.get(c['name'], c['name'])
        name = name.replace('_', ' ').strip().lower()
        if name in name_to_id:
            mapping[c['id']] = name_to_id[name]
    return mapping


def main():
    parser = argparse.ArgumentParser(description='AODRaw COCO → YOLO 格式转换')
    parser.add_argument('--ann', required=True, help='COCO 标注 JSON')
    parser.add_argument('--img-dir', required=True, help='图像目录（降采样 sRGB JPG）')
    parser.add_argument('--out', required=True, help='输出目录（datasets/aodraw）')
    parser.add_argument('--img-ext', default='.JPG', help='图像扩展名')
    args = parser.parse_args()

    ann = json.load(open(args.ann, encoding='utf-8'))
    class_map = build_class_map(ann['categories'])
    print(f'类别映射: {len(class_map)}/63 个 AODRaw 类别可映射到 COCO')

    anns_by_img = defaultdict(list)
    for a in ann['annotations']:
        anns_by_img[a['image_id']].append(a)

    out = Path(args.out)
    (out / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    img_dir = Path(args.img_dir).resolve()

    n_img = n_label = 0
    for img in ann['images']:
        w, h = img['width'], img['height']
        lines = []
        for a in anns_by_img[img['id']]:
            cid = class_map.get(a['category_id'])
            if cid is None:
                continue
            x, y, bw, bh = a['bbox']
            if bw <= 0 or bh <= 0:
                continue
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw, nh = bw / w, bh / h
            if not (0 < cx < 1 and 0 < cy < 1):
                continue
            lines.append(f'{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}')
        if lines:
            stem = Path(img['file_name']).stem
            (out / 'labels' / 'train' / f'{stem}.txt').write_text('\n'.join(lines))
            n_label += 1
        n_img += 1

    # 图像目录符号链接（不复制 4.3G 数据）
    (out / 'images').mkdir(parents=True, exist_ok=True)
    link = out / 'images' / 'train'
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(img_dir, target_is_directory=True)

    # data.yaml（names 为映射后的类别）
    names = {cid: COCO_NAMES[cid] for cid in sorted(set(class_map.values()))}
    yaml_lines = [
        f'path: {out.resolve()}',
        'train: images/train',
        'val: images/train',
        '',
        f'nc: {len(names)}',
        'names:',
    ]
    for cid, name in sorted(names.items()):
        yaml_lines.append(f'  {cid}: {name}')
    (out / 'data.yaml').write_text('\n'.join(yaml_lines), encoding='utf-8')

    print(f'图像总数: {n_img}, 生成标签文件: {n_label}, 类别数: {len(names)}')
    print(f'输出: {out.resolve()} (images/train 为符号链接, data.yaml 已生成)')


if __name__ == '__main__':
    main()
