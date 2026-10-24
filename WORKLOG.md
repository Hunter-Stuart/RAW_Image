# 工作日志

## 2026-10-24 仓库重建记录
- 本地 .git 目录（24 个提交的历史）因误删丢失，GitHub 远程仅有 V0.0（main 分支）。
- 重建方案：以当前工作树（代码完整）为准重新初始化仓库，README/LICENSE 从 GitHub main 恢复，
  全部成果作为新快照提交到 feature/fix-core-bugs 分支。
- 工作树内全部 2D 阶段成果完整保留：10 个脚本、raw_source/adaptive_params 两个模块、
  实验图表与数据（binocular_isp/data/output/benchmark/）、best_v2.pt 等。

## 2D 阶段成果摘要（详细见 D:\firstwork\RAW_Image_实验报告.md）
- 7 个 bug 修复；KITTI 校正验证 0.355px；AODRaw 真实 RAW 验证。
- ISP vs 相机直出基准；参数扫描；场景自适应模块（adaptive_params.py）+ 三方对比验证。
- v1/v2 训练：全验证集 mAP50 0.327 → 0.416（+0.089），四条件全提升。
- 暗光专项：提亮1档+禁锐化最优（+0.033）；雨天需锐化——场景参数方向相反的铁证。
- 方法论：全量评测为唯一报告口径（样本漂移 0.50→0.13 实例）。

## 下一步
- 3D 检测准备：KITTI 3D 数据、视差→深度 demo、双目 3D 基线复现、ISP×3D 实验。
- 待 FLIR 相机到货真机验证。
