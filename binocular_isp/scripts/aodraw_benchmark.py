#!/usr/bin/env python
"""
AODRaw 条件对比基准测试

对比"本项目 ISP 管线（改进算法）"与"相机直出 JPG（常规算法）"在
不同拍摄条件（晴天/暗光/雾天/雨天）下的表现：
  1. 图像质量指标: 亮度/对比度/清晰度/噪声 + PSNR/SSIM（以相机JPG为参考）
  2. 目标检测指标: 按条件统计 mAP50（YOLOv8n 在两种图像上的检测对比）
  3. 可视化: 指标柱状图 + 检测结果并排图格

用法:
  python scripts/aodraw_benchmark.py [--per-cond 10] [--zip AODRaw_test.zip]

说明:
  - AODRaw 无"阴天"标签，其 9 种条件为 光照(白天/暗光)×天气(晴/雨/雾)+室内外，
    阴天用雨天(rain)近似
  - YOLOv8n 为 COCO 预训练权重，仅在与 COCO 类别同名(≈30/63类)的类别上评测
  - 输出: data/output/benchmark/
"""

import argparse
import json
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from isp_pipeline.pipeline import ISPPipeline

# 条件分组: 显示名 -> (标签集合, 每条件图像数)
COND_GROUPS = {
    '晴天': {'tags': ('normal_light',), 'n': 10},
    '暗光': {'tags': ('low_light',), 'n': 10},
    '雾天': {'tags': ('fog',), 'n': 10},
    '雨天': {'tags': ('rain',), 'n': 10},
}

# AODRaw 类别名 → COCO 类别名 的显式修正
NAME_OVERRIDES = {'bicyle': 'bicycle', 'phone': 'cell phone'}


def build_class_map(ann_categories, yolo_names):
    """AODRaw类别id -> COCO类别id（按名称匹配）"""
    coco_name_to_id = {v: k for k, v in yolo_names.items()}
    mapping = {}
    for c in ann_categories:
        name = NAME_OVERRIDES.get(c['name'], c['name'])
        name = name.replace('_', ' ').strip().lower()
        if name in coco_name_to_id:
            mapping[c['id']] = coco_name_to_id[name]
    return mapping


def box_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (ua + 1e-9)


def compute_ap50(dets, gts):
    """VOC 风格 11 点插值 AP50。dets: [(score,x1,y1,x2,y2)]"""
    if not gts:
        return 0.0 if not dets else 0.0
    dets = sorted(dets, key=lambda d: -d[0])
    tp = np.zeros(len(dets)); fp = np.zeros(len(dets))
    matched = np.zeros(len(gts), bool)
    for i, d in enumerate(dets):
        best, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if matched[j]:
                continue
            iou = box_iou(d[1:], g)
            if iou > best:
                best, best_j = iou, j
        if best >= 0.5 and best_j >= 0:
            tp[i] = 1; matched[best_j] = True
        else:
            fp[i] = 1
    tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
    rec = tp_c / len(gts)
    prec = tp_c / np.maximum(tp_c + fp_c, 1)
    ap = 0.0
    for t in np.arange(0, 1.01, 0.1):
        ap += float(prec[rec >= t].max()) / 11 if (rec >= t).any() else 0.0
    return ap


def iq_metrics(img):
    """无参考图像质量指标"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    residual = gray - cv2.GaussianBlur(gray, (3, 3), 0)
    return {
        'brightness': float(gray.mean()),
        'contrast': float(gray.std()),
        'sharpness': float(cv2.Laplacian(gray, cv2.CV_32F).var()),
        'noise': float(residual.std()),
    }


def psnr_ssim(a, b):
    """灰度图 PSNR + SSIM（避免色彩差异主导，参考 b=相机JPG）"""
    ps = cv2.PSNR(a, b)
    ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32), \
             cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mu = cv2.GaussianBlur(ga, (11, 11), 1.5)
    mv = cv2.GaussianBlur(gb, (11, 11), 1.5)
    m2, v2, muv = mu * mu, mv * mv, mu * mv
    sx = cv2.GaussianBlur(ga * ga, (11, 11), 1.5) - m2
    sy = cv2.GaussianBlur(gb * gb, (11, 11), 1.5) - v2
    sxy = cv2.GaussianBlur(ga * gb, (11, 11), 1.5) - muv
    s1 = (2 * muv + 6.5) / (m2 + v2 + 6.5)
    s2 = (2 * sxy + 58.5) / (sx + sy + 58.5)
    return ps, float((s1 * s2).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-cond', type=int, default=10, help='每条件图像数')
    parser.add_argument('--exposure-bias', type=float, default=0.0,
                        help='管线曝光补偿(stop)，输出=图像x2^bias，调优实验用')
    parser.add_argument('--contrast', type=float, default=1.0,
                        help='管线对比度(色调映射contrast参数)，调优实验用')
    parser.add_argument('--sharpen-strength', type=float, default=0.0,
                        help='管线unsharp锐化强度(>0开启)，调优实验用')
    parser.add_argument('--saturation', type=float, default=1.0,
                        help='色调映射饱和度(晴天调优用)')
    parser.add_argument('--tone-method', default='linear', choices=['linear', 'filmic'],
                        help='色调映射方式: linear=标准伽马(默认), filmic=ACES风格S曲线')
    parser.add_argument('--zip', default=str(project_root.parent / 'AODRaw_test.zip'))
    parser.add_argument('--ann', default=str(project_root.parent / 'AODRaw_annotations/AODRaw/annotations/test_annotations.json'))
    parser.add_argument('--weights', default=str(project_root / 'data/yolov8n.pt'))
    args = parser.parse_args()

    out = project_root / 'data' / 'output' / 'benchmark'
    work = project_root / 'data' / 'input' / 'aodraw_bench'
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 读标注，按条件选图 ----------
    print('[1] 读取标注并按条件选图...')
    ann = json.load(open(args.ann, encoding='utf-8'))
    cat_names = {c['id']: c['name'] for c in ann['categories']}
    img_meta = {img['id']: img for img in ann['images']}
    gts_by_img = defaultdict(list)
    for a in ann['annotations']:
        gts_by_img[a['image_id']].append(a)

    from ultralytics import YOLO
    model = YOLO(args.weights)
    class_map = build_class_map(ann['categories'], model.names)
    print(f'  类别映射: {len(class_map)}/63 个 AODRaw 类别与 COCO 同名')

    selected = {}
    for cond, cfg in COND_GROUPS.items():
        cfg['n'] = args.per_cond
        picked = []
        for img in ann['images']:
            tags = tuple(sorted(img.get('tag', [])))
            if any(t in cfg['tags'] for t in tags):
                # 只选有"可映射类别"标注的图像
                anns = gts_by_img[img['id']]
                if any(a['category_id'] in class_map for a in anns):
                    picked.append(img)
                    if len(picked) >= cfg['n']:
                        break
        selected[cond] = picked
        print(f'  {cond}: 选中 {len(picked)} 张')

    # ---------- 2. 从 zip 提取对应 ARW+JPG（zip 缺失时依赖已提取文件） ----------
    print('[2] 提取图像...')
    zf = zipfile.ZipFile(args.zip) if Path(args.zip).exists() else None
    members = set(zf.namelist()) if zf else set()
    if zf is None:
        print('  ! 未找到测试zip, 使用 work 目录中已提取的 ARW/JPG')
    pipeline = ISPPipeline({'isp': {'srgb': {'exposure_bias': args.exposure_bias,
                                             'contrast': args.contrast,
                                             'saturation': args.saturation,
                                             'tone_method': args.tone_method},
                                    'sharpen': {'method': 'unsharp' if args.sharpen_strength > 0 else 'none',
                                                'strength': args.sharpen_strength}},
                            'output': {'width': 1280, 'height': 720, 'bit_depth': 8}})

    metrics = {}
    for cond, imgs in selected.items():
        m = {'iq': {'A': defaultdict(list), 'B': defaultdict(list), 'B2': defaultdict(list)},
             'psnr': [], 'ssim': [], 'psnr_b2': [], 'ssim_b2': [],
             'det': {'A': defaultdict(list), 'B': defaultdict(list), 'B2': defaultdict(list)},
             'gt': defaultdict(list)}
        for img in imgs:
            stem = Path(img['file_name']).stem
            for ext in ('.ARW', '.JPG'):
                member = f'AODRaw/images/{stem}{ext}'
                if zf is not None and member in members and not (work / member).exists():
                    zf.extract(member, work)
            raw_path = work / 'AODRaw/images' / f'{stem}.ARW'
            jpg_path = work / 'AODRaw/images' / f'{stem}.JPG'
            if not raw_path.exists() or not jpg_path.exists():
                print(f'  ! 缺少文件: {stem}')
                continue

            # A路: 本项目管线 (改进算法)，输出缓存以便重跑时复用
            cache_a = work / f'A_b{args.exposure_bias}c{args.contrast}s{args.sharpen_strength}m{args.tone_method}e{args.exposure_bias}c{args.contrast}sat{args.saturation}' / f'{stem}.png'
            cache_a.parent.mkdir(parents=True, exist_ok=True)
            if cache_a.exists():
                img_a = cv2.imread(str(cache_a))
            else:
                t0 = time.time()
                img_a, _ = pipeline.process_file(str(raw_path))
                cv2.imwrite(str(cache_a), img_a)
            # B路: 相机直出 JPG (常规算法)
            img_b = cv2.resize(cv2.imread(str(jpg_path)), (1280, 720))
            # B2路: rawpy 默认后处理管线（LibRaw 经典开源 RAW 处理器）
            import rawpy
            with rawpy.imread(str(raw_path)) as raw:
                post = raw.postprocess(use_camera_wb=True,
                                       output_color=rawpy.ColorSpace.sRGB, output_bps=8)
            img_b2 = cv2.resize(cv2.cvtColor(post, cv2.COLOR_RGB2BGR), (1280, 720))

            # 图像质量指标
            iqa, iqb, iqb2 = iq_metrics(img_a), iq_metrics(img_b), iq_metrics(img_b2)
            for k, v in iqa.items():
                m['iq']['A'][k].append(v)
            for k, v in iqb.items():
                m['iq']['B'][k].append(v)
            for k, v in iqb2.items():
                m['iq']['B2'][k].append(v)
            ps, ss = psnr_ssim(img_a, img_b)
            m['psnr'].append(ps); m['ssim'].append(ss)
            ps2, ss2 = psnr_ssim(img_b2, img_b)
            m['psnr_b2'].append(ps2); m['ssim_b2'].append(ss2)

            # 检测（A/B/B2 各跑一次）
            for side, im in (('A', img_a), ('B', img_b), ('B2', img_b2)):
                r = model.predict(im, conf=0.05, verbose=False)[0]
                boxes = r.boxes
                if boxes is not None and len(boxes):
                    xyxy = boxes.xyxy.cpu().numpy()
                    scores = boxes.conf.cpu().numpy()
                    cls = boxes.cls.cpu().numpy().astype(int)
                    for bb, s, c in zip(xyxy, scores, cls):
                        m['det'][side][c].append((float(s), *bb.tolist()))

            # 真值（映射后的 COCO 类别 id）。
            # 注意: 标注 bbox 是原图(6000x4000)坐标，需缩放到 1280x720 检测空间
            ow = img.get('width', 6000)
            oh = img.get('height', 4000)
            sx, sy = 1280.0 / ow, 720.0 / oh
            for a in gts_by_img[img['id']]:
                coco_id = class_map.get(a['category_id'])
                if coco_id is None:
                    continue
                bb = a['bbox']  # COCO xywh
                m['gt'][coco_id].append((bb[0] * sx, bb[1] * sy,
                                         (bb[0] + bb[2]) * sx, (bb[1] + bb[3]) * sy))

            print(f'  {cond} {stem}: PSNR={ps:.1f}dB SSIM={ss:.3f}')
        metrics[cond] = m

    # ---------- 3. 汇总统计 ----------
    print('[3] 汇总指标...')
    summary = {}
    for cond, m in metrics.items():
        iq = {side: {k: float(np.mean(v)) if v else 0.0
                     for k, v in m['iq'][side].items()} for side in ('A', 'B', 'B2')}
        ap = {}
        for side in ('A', 'B', 'B2'):
            aps = [compute_ap50(m['det'][side].get(c, []), gts)
                   for c, gts in m['gt'].items() if gts]
            ap[side] = float(np.mean(aps)) if aps else 0.0
        summary[cond] = {
            'iq_A': iq['A'], 'iq_B': iq['B'], 'iq_B2': iq['B2'],
            'psnr_mean': float(np.mean(m['psnr'])) if m['psnr'] else 0.0,
            'ssim_mean': float(np.mean(m['ssim'])) if m['ssim'] else 0.0,
            'psnr_b2_mean': float(np.mean(m['psnr_b2'])) if m['psnr_b2'] else 0.0,
            'ssim_b2_mean': float(np.mean(m['ssim_b2'])) if m['ssim_b2'] else 0.0,
            'mAP50_A': ap['A'], 'mAP50_B': ap['B'], 'mAP50_B2': ap['B2'],
            'n_images': len(m['psnr']),
        }
        s = summary[cond]
        print(f"\n  === {cond} ({s['n_images']}张) ===")
        print(f"    亮度  A={s['iq_A']['brightness']:.1f} B={s['iq_B']['brightness']:.1f} B2={s['iq_B2']['brightness']:.1f}"
              f" | 清晰度 A={s['iq_A']['sharpness']:.1f} B={s['iq_B']['sharpness']:.1f} B2={s['iq_B2']['sharpness']:.1f}")
        print(f"    PSNR  A={s['psnr_mean']:.1f}dB B2={s['psnr_b2_mean']:.1f}dB | SSIM A={s['ssim_mean']:.3f} B2={s['ssim_b2_mean']:.3f}")
        print(f"    mAP50  A(改进)={s['mAP50_A']:.3f}  B(相机JPG)={s['mAP50_B']:.3f}  B2(rawpy默认)={s['mAP50_B2']:.3f}")

    json.dump(summary, open(out / 'metrics.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)

    # ---------- 4. 可视化 ----------
    print('\n[4] 生成可视化图表...')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    conds = list(summary.keys())
    x = np.arange(len(conds))
    width = 0.35

    # 图1: 图像质量指标 2x2（三方对比）
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, key, title in zip(axes.flat,
                              ['brightness', 'contrast', 'sharpness', 'noise'],
                              ['亮度(均值)', '对比度(标准差)', '清晰度(Laplacian方差)', '噪声(高频残差std)']):
        va = [summary[c]['iq_A'][key] for c in conds]
        vb = [summary[c]['iq_B'][key] for c in conds]
        vb2 = [summary[c]['iq_B2'][key] for c in conds]
        w = 0.27
        ax.bar(x - w, va, w, label='自研管线(A)')
        ax.bar(x, vb, w, label='相机JPG(B)')
        ax.bar(x + w, vb2, w, label='rawpy默认(B2)')
        ax.set_xticks(x); ax.set_xticklabels(conds); ax.set_title(title); ax.legend(fontsize=8)
    fig.suptitle('三方对比: 自研管线 vs 相机JPG vs rawpy默认', fontsize=14)
    fig.tight_layout()
    fig.savefig(out / 'chart_iq.png', dpi=120)
    plt.close(fig)

    # 图2: 检测 mAP50 + PSNR/SSIM（三方）
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    va = [summary[c]['mAP50_A'] for c in conds]
    vb = [summary[c]['mAP50_B'] for c in conds]
    vb2 = [summary[c]['mAP50_B2'] for c in conds]
    w = 0.27
    axes[0].bar(x - w, va, w, label='自研管线(A)')
    axes[0].bar(x, vb, w, label='相机JPG(B)')
    axes[0].bar(x + w, vb2, w, label='rawpy默认(B2)')
    axes[0].set_xticks(x); axes[0].set_xticklabels(conds)
    axes[0].set_title('检测 mAP50 对比'); axes[0].legend(fontsize=8)
    vp = [summary[c]['psnr_mean'] for c in conds]
    vp2 = [summary[c]['psnr_b2_mean'] for c in conds]
    vs = [summary[c]['ssim_mean'] for c in conds]
    vs2 = [summary[c]['ssim_b2_mean'] for c in conds]
    axes[1].bar(x - w, vp, w, label='PSNR A(dB)')
    axes[1].bar(x, vp2, w, label='PSNR B2(dB)')
    axes[1].plot(x, vs, 'o-', color='green', label='SSIM A')
    axes[1].plot(x, vs2, 's--', color='orange', label='SSIM B2')
    axes[1].set_xticks(x); axes[1].set_xticklabels(conds)
    axes[1].set_title('与相机JPG的相似度'); axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'chart_detection.png', dpi=120)
    plt.close(fig)

    # 图3: 每条件选1张的并排检测对比图格（OpenCV putText 不支持中文，用英文标注）
    cond_en = {'晴天': 'sunny', '暗光': 'low-light', '雾天': 'foggy', '雨天': 'rainy'}
    rows = []
    for cond in conds:
        img = selected[cond][0]
        stem = Path(img['file_name']).stem
        raw_path = work / 'AODRaw/images' / f'{stem}.ARW'
        jpg_path = work / 'AODRaw/images' / f'{stem}.JPG'
        im_a, _ = pipeline.process_file(str(raw_path))
        im_b = cv2.resize(cv2.imread(str(jpg_path)), (1280, 720))
        ra = model.predict(im_a, conf=0.25, verbose=False)[0].plot()
        rb = model.predict(im_b, conf=0.25, verbose=False)[0].plot()
        pair = np.vstack([cv2.putText(rb.copy(), f'{cond_en[cond]} | camera JPG (baseline)', (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2),
                          cv2.putText(ra.copy(), f'{cond_en[cond]} | our ISP pipeline', (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)])
        rows.append(pair)
    grid = np.hstack(rows)
    cv2.imwrite(str(out / 'grid_side_by_side.png'), grid)
    print(f'  图表已保存: {out}')
    print('完成')


if __name__ == '__main__':
    main()
