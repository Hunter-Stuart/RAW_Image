#!/usr/bin/env python
"""
四方 ISP 对比可视化（补做）：
从 metrics_对齐后.json 与 cmp4_对齐前.log 生成:
  1. fourway_mAP50.png   四方法×四条件分组柱状图（对齐后）
  2. taisp_align.png     TA-ISP 对齐前后对照柱状图
  3. fourway_table.csv   数值表（可直接粘贴进报告）

用法: python scripts/plot_fourway.py
"""

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parent.parent
out_dir = project_root / 'data/output/benchmark'
summary_dir = Path('D:/firstwork/RAW_Image成果汇总/服务器产物')

CONDS = ['晴天', '暗光', '雾天', '雨天']
METHODS = [('A', '自研管线'), ('B', '相机JPG'), ('B2', 'rawpy默认'), ('B3', 'TA-ISP')]


def load_aligned():
    m = json.load(open(summary_dir / 'metrics_对齐后.json', encoding='utf-8'))
    return {c: {k: m[c][f'mAP50_{k}'] for k, _ in METHODS} for c in CONDS}


def load_unaligned_b3():
    txt = open(summary_dir / 'cmp4_对齐前.log', encoding='utf-8', errors='ignore').read()
    vals = {}
    for c in CONDS:
        for line in txt.splitlines():
            if line.strip().startswith('mAP50') and f'{c}' in txt.splitlines()[txt.splitlines().index(line) - 1] if False else False:
                pass
    # 简单解析: 找含 "B3(TA-ISP)=" 的行, 按条件顺序
    b3_lines = [l for l in txt.splitlines() if 'B3(TA-ISP)=' in l]
    for c, l in zip(CONDS, b3_lines):
        vals[c] = float(re.search(r'B3\(TA-ISP\)=([\d.]+)', l).group(1))
    return vals


def main():
    aligned = load_aligned()
    unaligned_b3 = load_unaligned_b3()

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 图1: 四方法 × 四条件（对齐后）
    x = np.arange(len(CONDS))
    w = 0.2
    colors = ['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e']
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (key, label) in enumerate(METHODS):
        vals = [aligned[c][key] for c in CONDS]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=label, color=colors[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.008, f'{v:.3f}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(CONDS)
    ax.set_ylabel('mAP50'); ax.set_title('四方 ISP 对比（TA-ISP 为预处理对齐后）')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'fourway_mAP50.png', dpi=130)
    plt.close(fig)

    # 图2: TA-ISP 对齐前后
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(CONDS))
    b_old = [unaligned_b3[c] for c in CONDS]
    b_new = [aligned[c]['B3'] for c in CONDS]
    ax.bar(x - 0.2, b_old, 0.4, label='对齐前(rawpy后处理输入)', color='#1f77b4')
    ax.bar(x + 0.2, b_new, 0.4, label='对齐后(官方Debayer输入)', color='#ff7f0e')
    for xi, vo, vn in zip(x, b_old, b_new):
        ax.text(xi - 0.2, vo + 0.008, f'{vo:.3f}', ha='center', fontsize=9)
        ax.text(xi + 0.2, vn + 0.008, f'{vn:.3f}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(CONDS)
    ax.set_ylabel('mAP50'); ax.set_title('TA-ISP 预处理对齐前后对比')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'taisp_align.png', dpi=130)
    plt.close(fig)

    # CSV 表
    with open(out_dir / 'fourway_table.csv', 'w', encoding='utf-8') as f:
        f.write('条件,' + ','.join(l for _, l in METHODS) + '\n')
        for c in CONDS:
            f.write(c + ',' + ','.join(f"{aligned[c][k]:.3f}" for k, _ in METHODS) + '\n')
        f.write('\nTA-ISP对齐前,' + ','.join('' for _ in range(3)) +
                ','.join(f'{unaligned_b3[c]:.3f}' for c in CONDS) + '\n')

    print('产出:')
    print(f'  {out_dir}/fourway_mAP50.png   (四方法×四条件柱状图)')
    print(f'  {out_dir}/taisp_align.png     (TA-ISP对齐前后对照)')
    print(f'  {out_dir}/fourway_table.csv   (数值表)')
    print('\n数值表:')
    print('  条件      ' + '  '.join(f'{l:>10}' for _, l in METHODS))
    for c in CONDS:
        print(f'  {c}  ' + '  '.join(f'{aligned[c][k]:>10.3f}' for k, _ in METHODS))


if __name__ == '__main__':
    main()
