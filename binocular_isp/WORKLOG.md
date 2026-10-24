# 工作日志

## 2026-09-21
- 环境验证通过（4/4 测试通过，标定校验通过）。
- 创建 feature/fix-core-bugs 分支，准备修复极线校正硬编码问题。
- **【已完成】修复 run_pipeline.py 极线校正硬编码 Bug，成功动态读取 stereo_calibration.yaml 外参。**
## 2026-09-28
- AODRaw 真实RAW链路验证通过（3图对比同量级），修复rawpy 0.27三个兼容bug。
- KITTI 真实数据集端到端验证通过，垂直误差 0.355px（<0.5px 达标）。