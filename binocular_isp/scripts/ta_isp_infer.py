#!/usr/bin/env python
"""
TA-ISP 推理适配器：AODRaw ARW → 去马赛克 → TA-ISP → RGB 输出
"""

import argparse
import inspect
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
torch.backends.cudnn.enabled = False  # 兼容 cuDNN 版本错配导致的段错误

sys.path.insert(0, '/home/zeng/TA-ISP/RAOD')
from models.taisp import MultipleMaskISP_Conv


def load_isp_weights(ckpt_path):
    """从 YOLOX+TA-ISP 联合检查点中提取 ISP 子模块权重（防御式）"""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    if 'state_dict' in sd:
        sd = sd['state_dict']
    # 过滤出 ISP 相关键（dgain/attn/mask 等，排除 yolox 骨干）
    isp_keys = [k for k in sd if any(t in k.lower() for t in ('isp', 'dgain', 'mask', 'attn'))]
    if not isp_keys:
        print('! 未找到 ISP 键，检查点前 30 个键:')
        for k in list(sd)[:30]:
            print('   ', k)
        sys.exit(1)
    prefix = os.path.commonprefix(isp_keys)
    sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    print(f'  ISP 子模块: {len(sub)} 个参数, 前缀="{prefix}"')
    print('  示例键:', list(sub)[:6])
    return sub


def build_model(sub_sd):
    """按检查点形状构造 MultipleMaskISP_Conv（防御式，多级回退）"""
    first_shape = next(iter(sub_sd.values())).shape
    print(f'  首参数 shape: {list(first_shape)}')
    in_ch = 3  # 强制为 3 通道
    for args in [(in_ch,), (3,), ()]:
        try:
            m = MultipleMaskISP_Conv(*args)
            m.load_state_dict(sub_sd, strict=False)
            print(f'  构造成功: MultipleMaskISP_Conv(*{args})')
            return m.eval()
        except Exception as e:
            print(f'  构造尝试 {args} 失败: {type(e).__name__}: {e}')
    print('  构造签名:', inspect.signature(MultipleMaskISP_Conv.__init__))
    sys.exit(1)


def preprocess_arw(arw_path):
    """ARW → 640×640×3 fp32 [0,1]（使用 rawpy 安全去马赛克）"""
    import rawpy
    print(f"    - 正在使用 rawpy 读取 ...", flush=True)
    try:
        with rawpy.imread(str(arw_path)) as raw:
            print(f"    - 读取成功，正在去马赛克 ...", flush=True)
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8)
    except Exception as e:
        print(f"    - ❌ [错误] 读取/去马赛克失败: {e}", flush=True)
        return None

    try:
        print(f"    - 正在缩放和转 Tensor ...", flush=True)
        rgb = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
        rgb_cont = np.ascontiguousarray(rgb.astype(np.float32) / 255.0)
        x = torch.from_numpy(rgb_cont).permute(2, 0, 1).unsqueeze(0)
        return x
    except Exception as e:
        print(f"    - ❌ [错误] 缩放/转Tensor失败: {e}", flush=True)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arw-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--ann', required=True, help='test_annotations.json（判断暗光）')
    parser.add_argument('--ckpt-day', required=True)
    parser.add_argument('--ckpt-night', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dry-run', action='store_true', help='只探测检查点结构')
    args = parser.parse_args()

    sub_day = load_isp_weights(args.ckpt_day)
    if args.dry_run:
        build_model(sub_day)
        print('dry-run 完成')
        return
    model_day = build_model(sub_day).to(args.device)
    sub_night = load_isp_weights(args.ckpt_night)
    model_night = build_model(sub_night).to(args.device)

    import json
    ann = json.load(open(args.ann, encoding='utf-8'))
    dark_stems = {Path(i['file_name']).stem for i in ann['images'] if 'low_light' in i.get('tag', [])}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arws = sorted(Path(args.arw_dir).glob('*.ARW'))
    print(f'共 {len(arws)} 张 ARW')
    
    for i, p in enumerate(arws):
        stem = p.stem
        print(f'[{i+1}/{len(arws)}] 正在处理: {stem} ...', flush=True)
        
        x = preprocess_arw(p)
        if x is None:
            print(f'  [跳过] {stem} 预处理失败', flush=True)
            continue
            
        x = x.to(args.device)
        model = model_night if stem in dark_stems else model_day
        
        try:
            with torch.no_grad():
                im = model(x)
                
            im = im.squeeze(0).permute(1, 2, 0).cpu().numpy()
            im = np.clip(im, 0, 1)
            img = (im * 255).astype(np.uint8)
            img = cv2.resize(img, (1280, 720))
            cv2.imwrite(str(out / f'{stem}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f'  ❌ [错误] 模型推理或保存失败: {e}', flush=True)
        finally:
            # 无论成功失败，强制清理内存，防止 OOM
            if 'im' in locals(): del im
            if 'img' in locals(): del img
            if 'x' in locals(): del x
            import gc; gc.process() if hasattr(gc, 'process') else gc.collect()
            print(f'  -> {stem} 处理完毕，内存已清理', flush=True)


if __name__ == '__main__':
    main()