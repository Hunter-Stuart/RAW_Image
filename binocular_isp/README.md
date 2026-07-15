# E-SIP Binocular ISP Pipeline

面向**双目3D目标检测**下游任务的 RAW → sRGB 图像处理管线。

## 定位

本管线是 E-SIP 项目中的**图像处理中间环节**：

```
相机传感器 (RAW)
    ↓
┌─────────────────────────────────────┐
│        E-SIP ISP Pipeline           │
│  RAW读取 → 预处理 → 去马赛克 →      │
│  色彩校正 → Gamma → 去噪/锐化 →    │
│  去畸变/极线校正 → 质量检查         │
└─────────────────────────────────────┘
    ↓
sRGB 图像（左右图 + 元数据）
    ↓
下游: 双目立体匹配 → 视差/深度 → 3D目标检测
```

## 项目结构

```
binocular_isp/
├── configs/                    # 配置文件
│   ├── pipeline_config.yaml    # ISP 管线配置
│   ├── camera_left.yaml        # 左相机参数（占位）
│   ├── camera_right.yaml       # 右相机参数（占位）
│   └── stereo_calibration.yaml # 双目标定参数（占位）
├── src/                        # 源代码
│   ├── isp_pipeline/           # ISP 管线核心
│   │   ├── raw_reader.py       # RAW文件读取
│   │   ├── pre_process.py      # 黑电平校正、白平衡
│   │   ├── demosaic.py         # Bayer去马赛克
│   │   ├── color_correction.py # 色彩校正、Gamma、色调映射
│   │   ├── denoise_sharpen.py  # 去噪、锐化
│   │   └── pipeline.py         # Pipeline编排器
│   ├── binocular/              # 双目相关
│   │   ├── sync_control.py     # 曝光/ISP参数一致性控制
│   │   └── rectify.py          # 去畸变、极线校正
│   ├── config/                 # 配置管理
│   │   ├── camera_params.py    # 相机参数解析
│   │   └── isp_config.py       # ISP配置管理
│   └── utils/                  # 工具
│       ├── metadata.py         # 元数据管理
│       └── visualization.py    # 调试可视化
├── scripts/                    # 运行脚本
│   ├── run_pipeline.py         # 主运行脚本
│   └── batch_process.py        # 批量处理
├── tests/                      # 测试
│   └── test_pipeline.py        # 管线测试
├── data/                       # 数据目录
│   ├── input/                  # 输入 RAW 文件
│   └── output/                 # 输出 sRGB 图像
├── requirements.txt            # Python依赖
└── README.md                   # 本文件
```

## 安装

### 1. 安装 Python 依赖

```bash
pip install rawpy PyYAML tqdm matplotlib pytest -q
```

### 2. 替换占位配置

将 `configs/camera_left.yaml` 和 `configs/camera_right.yaml` 中的相机内参、畸变系数替换为**实际标定值**。

将 `configs/stereo_calibration.yaml` 中的双目标定参数替换为实际值。

## 使用

### 处理一对双目图像

```bash
python scripts/run_pipeline.py \
    --left data/input/left_0001.ARW \
    --right data/input/right_0001.ARW \
    --output data/output \
    --rectify \
    --check-consistency
```

### 批量处理

```bash
python scripts/run_pipeline.py \
    --left data/input/left_dir \
    --right data/input/right_dir \
    --output data/output \
    --rectify
```

或者使用专门的批处理脚本：

```bash
python scripts/batch_process.py \
    --left-dir data/input/left \
    --right-dir data/input/right \
    --output data/output
```

### 仅处理 RAW，不做校正

```bash
python scripts/run_pipeline.py --left L.ARW --right R.ARW --raw-only
```

## ISP 管线流程

| 步骤 | 模块 | 功能 | 可配置 |
|------|------|------|--------|
| 1 | `raw_reader` | 读取 RAW 文件，提取 Bayer 数据和元数据 | Bayer 模式 |
| 2 | `pre_process` | 黑电平校正、白平衡 | 黑/白电平值、WB增益 |
| 3 | `demosaic` | Bayer → RGB 插值 | 方法: opencv/bilinear/malvar |
| 4 | `color_correction` | 色彩空间转换、Gamma | 色彩矩阵、Gamma值 |
| 5 | `denoise_sharpen` | 可选去噪、锐化 | 方法、强度 |
| 6 | `pipeline` | 缩放、格式转换 | 目标分辨率、位深 |

## 双目一致性保障

- **ISP参数共享**：左右图使用相同的白平衡、Gamma等参数
- **曝光同步**：通过 `StereoConsistencyController` 统一左右相机控制参数
- **几何校正**：去畸变 + 极线校正，确保同名点在同一水平线上
- **质量检测**：`StereoConsistencyChecker` 检查亮度、颜色、清晰度一致性

## 输出

每对图像处理后会生成一个子目录，包含：

- `left.png` / `right.png` — 处理后的 sRGB 图像
- `metadata.json` — 全链路元数据（处理参数、校正质量等）

输出符合 README 下游需求：
- RGB888 (8bit/通道) 或 16bit
- sRGB 色彩空间
- PNG 无压缩格式
- 1280×720 分辨率（可配置）

## 与开源框架的关系

### [AODRaw](https://github.com/lzyhha/AODRaw) (CVPR 2025)
- 本项目的 RAW 读取模块参考了 AODRaw 的预处理代码
- AODRaw 的检测模型面向单目输入，本项目聚焦双目
- 待管线稳定后可对比 AODRaw 的 RAW-domain 检测方法

### [TA-ISP](https://github.com/CVL-UESTC/TA-ISP) (CVPR 2026)
- TA-ISP 的任务感知 ISP 思想是本管线"闭环反馈"阶段的参考方向
- 可参考其任务反馈机制，后续实现"根据下游检测结果自动调整ISP参数"

## 后续计划

- [ ] 集成 TA-ISP 风格的任务反馈闭环
- [ ] 添加 HDR 融合支持
- [ ] 在线质量监测仪表板
- [ ] ROS 接口集成
- [ ] 与 E-SIP Sensorkit 深度对接
