#!/usr/bin/env python
"""
TA-ISP 推理适配器（预处理对齐版）：AODRaw ARW → 官方Debayer3x3 → TA-ISP → RGB

预处理严格对齐 TA-ISP 官方 preprocess_raw.py：
  Bayer(归一化到0~255) → Debayer3x3(官方3x3卷积去马赛克, 分辨率减半)
  → 双线性缩放到640×640 → /255 归一化到[0,1] → 网络
（旧版用 rawpy.postprocess 是完整ISP，与训练输入域不符，此为对齐复测版）

权重选择: 暗光(low_light)用 night 权重，其余用 day 权重
用法（服务器 ispnet 环境）:
  python ta_isp_infer.py --arw-dir <dir> --out-dir <dir> --ann <json> \
      --ckpt-day <pth> --ckpt-night <pth> [--dry-run]
"""

import argparse
import gc
import inspect
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
torch.backends.cudnn.enabled = False  # 兼容 cuDNN 版本错配导致的段错误

sys.path.insert(0, '/home/zeng/TA-ISP/RAOD')
sys.path.insert(0, '/home/zeng/TA-ISP/RAOD/scripts')
from models.taisp import MultipleMaskISP_Conv
from preprocess_raw import Debayer3x3   # 官方去马赛克实现（对齐用）


def load_isp_weights(ckpt_path):
    """从 YOLOX+TA-ISP 联合检查点中提取 ISP 子模块权重"""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    if 'state_dict' in sd:
        sd = sd['state_dict']
    isp_keys = [k for k in sd if any(t in k.lower() for t in ('isp', 'dgain', 'mask', 'attn'))]
    if not isp_keys:
        print('! 未找到 ISP 键，检查点前 30 个键:')
        for k in list(sd)[:30]:
            print('   ', k)
        sys.exit(1)
    prefix = os.path.commonprefix(isp_keys)
    sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    print(f'  ISP 子模块: {len(sub)} 个参数, 前缀="{prefix}"')
    return sub


def build_model(sub_sd):
    first_shape = next(iter(sub_sd.values())).shape
    print(f'  首参数 shape: {list(first_shape)}')
    for args in [(3,), ()]:
        try:
            m = MultipleMaskISP_Conv(*args)
            m.load_state_dict(sub_sd, strict=False)
            print(f'  构造成功: MultipleMaskISP_Conv(*{args})')
            return m.eval()
        except Exception as e:
            print(f'  构造尝试 {args} 失败: {type(e).__name__}: {e}')
    print('  构造签名:', inspect.signature(MultipleMaskISP_Conv.__init__))
    sys.exit(1)


def preprocess_arw(arw_path, debayer):
    """对齐官方管线: Bayer(0~255) → Debayer3x3 → 640×640 → [0,1]"""
    import rawpy
    with rawpy.imread(str(arw_path)) as raw:
        bayer = raw.raw_image_visible.astype(np.float32)
        wl = raw.white_level or 16383
    bayer = np.clip(bayer / wl * 255.0, 0, 255)          # 与官方 clip 到 0~255 对齐
    x = torch.from_numpy(np.ascontiguousarray(bayer)).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    with torch.no_grad():
        rgb = debayer(x)                                 # (1,3,H/2,W/2)
    rgb = F.interpolate(rgb, size=(640, 640), mode='bilinear', align_corners=False)
    rgb = np.clip(rgb.squeeze(0).cpu().numpy(), 0, 255)
    return torch.from_numpy(np.ascontiguousarray(rgb / 255.0)).unsqueeze(0)     # (1,3,640,640) [0,1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arw-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--ann', required=True)
    parser.add_argument('--ckpt-day', required=True)
    parser.add_argument('--ckpt-night', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    sub_day = load_isp_weights(args.ckpt_day)
    if args.dry_run:
        build_model(sub_day)
        print('dry-run 完成')
        return
    model_day = build_model(sub_day).to(args.device)
    sub_night = load_isp_weights(args.ckpt_night)
    model_night = build_model(sub_night).to(args.device)
    debayer = Debayer3x3().to(args.device)

    ann = json.load(open(args.ann, encoding='utf-8'))
    dark_stems = {Path(i['file_name']).stem for i in ann['images']
                  if 'low_light' in i.get('tag', [])}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arws = sorted(Path(args.arw_dir).glob('*.ARW'))
    print(f'共 {len(arws)} 张 ARW')
    for i, p in enumerate(arws):
        stem = p.stem
        print(f'[{i+1}/{len(arws)}] {stem} ...', flush=True)
        try:
            x = preprocess_arw(p, debayer).to(args.device)
            model = model_night if stem in dark_stems else model_day
            with torch.no_grad():
                im = model(x)
            im = im.squeeze(0).permute(1, 2, 0).cpu().numpy()
            img = (np.clip(im, 0, 1) * 255).astype(np.uint8)
            img = cv2.resize(img, (1280, 720))
            cv2.imwrite(str(out / f'{stem}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f'  ❌ {stem} 失败: {e}', flush=True)
        finally:
            del x
            gc.collect()
    print(f'完成: {out}')


if __name__ == '__main__':
    main()
