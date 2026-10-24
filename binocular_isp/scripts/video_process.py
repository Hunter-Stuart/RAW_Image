#!/usr/bin/env python
"""
视频/帧序列处理脚本
功能：批量处理双目 RAW 帧序列（或视频文件），完成 ISP + 极线校正，
     输出 sRGB 演示视频（左右拼接或独立输出）。

输入支持三种模式（自动识别）：
  1) 帧序列目录：左右目录各放 .ARW/.DNG/.raw 等（真实相机 RAW）
  2) 帧序列目录：左右目录各放 .png（Bayer 阵列存成 PNG，调试用）
  3) 视频文件：灰度视频按 Bayer 流处理；彩色视频自动用 simulate_bayer_raw 模拟（demo 模式）

用法示例：
  # 真实 RAW 帧序列（拿到学长数据后）
  python scripts/video_process.py --left data/input/left --right data/input/right \
      --output data/output_video --fps 20

  # demo 模式：用任意彩色视频模拟双目，验证整条链路（无需真实相机数据）
  python scripts/video_process.py --left demo.mp4 --right demo.mp4 \
      --output data/output_video --no-rectify --max-frames 30
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from isp_pipeline.pipeline import ISPPipeline
from isp_pipeline.raw_reader import simulate_bayer_raw
from binocular.rectify import StereoRectifier, StereoCalibration
from binocular.sync_control import StereoConsistencyChecker
from config.camera_params import ConfigLoader

RAW_EXTS = ('.arw', '.dng', '.raw', '.cr2', '.nef')


def parse_args():
    parser = argparse.ArgumentParser(description="双目视频/帧序列 ISP 处理")
    parser.add_argument('--left', required=True, help='左路：帧序列目录 或 视频文件')
    parser.add_argument('--right', required=True, help='右路：帧序列目录 或 视频文件')
    parser.add_argument('--output', default='data/output_video', help='输出目录 (默认: data/output_video)')
    parser.add_argument('--config', default='pipeline_config.yaml', help='管线配置文件（相对 configs/ 目录）')
    parser.add_argument('--cam-left', default='camera_left.yaml', help='左相机参数文件（相对 configs/ 目录）')
    parser.add_argument('--cam-right', default='camera_right.yaml', help='右相机参数文件（相对 configs/ 目录）')
    parser.add_argument('--no-rectify', action='store_true', help='跳过极线校正（标定参数未就绪时用）')
    parser.add_argument('--check-consistency', action='store_true', help='每帧检查左右一致性')
    parser.add_argument('--layout', choices=['side_by_side', 'separate', 'both'], default='side_by_side',
                        help='视频输出布局: side_by_side=左右拼接(默认) separate=左右独立 both=都要')
    parser.add_argument('--fps', type=float, default=20.0, help='输出视频帧率（目录模式使用）')
    parser.add_argument('--max-frames', type=int, default=0, help='最多处理 N 帧 (0=全部, 调试用)')
    parser.add_argument('--save-frames', action='store_true', help='同时保存每帧 PNG 到 output/frames/')
    parser.add_argument('--white-level', type=int, default=None,
                        help='Bayer PNG/视频模式的饱和值（默认按位深自动: uint8→255, uint16→16383）')
    parser.add_argument('--bayer-pattern', default='RGGB', help='Bayer 排列 (默认: RGGB)')
    parser.add_argument('--detect', type=str, default=None,
                        help='YOLO 检测模型路径（如 yolov8n.pt），None=不检测')
    parser.add_argument('--detect-side', choices=['left', 'right', 'both'], default='both',
                        help='对哪路图像做检测 (默认: both)')
    parser.add_argument('--detect-conf', type=float, default=0.25, help='检测置信度阈值 (默认: 0.25)')
    return parser.parse_args()


def auto_white_level(bayer: np.ndarray, args) -> int:
    if args.white_level is not None:
        return args.white_level
    return 255 if bayer.dtype == np.uint8 else 16383


def build_rectifier(loader: ConfigLoader, args, frame_size) -> "StereoRectifier|None":
    """构建校正器。校正映射表只计算一次，后续帧直接 remap。"""
    left_cam = loader.load_camera_config('left', args.cam_left)
    right_cam = loader.load_camera_config('right', args.cam_right)
    stereo_config = loader.load('stereo_calibration.yaml')
    ext = stereo_config.get('stereo_extrinsics', {})

    calib = StereoCalibration(
        left_camera_matrix=loader.config_to_camera_matrix(left_cam),
        left_dist_coeffs=loader.config_to_dist_coeffs(left_cam),
        right_camera_matrix=loader.config_to_camera_matrix(right_cam),
        right_dist_coeffs=loader.config_to_dist_coeffs(right_cam),
        rotation_matrix=ConfigLoader.config_to_rotation(ext),
        translation_vector=ConfigLoader.config_to_translation(ext),
        image_size=frame_size,
    )
    if not calib.validate():
        print("  ! 标定参数不完整，跳过极线校正")
        return None

    rectifier = StereoRectifier(calib)
    rectifier.compute_rectification_maps(new_size=frame_size)
    print(f"  校正器就绪: T={calib.translation_vector.flatten()}")
    return rectifier


def load_dir_frames(dir_path: Path):
    """帧序列目录：先找 RAW，再找 Bayer PNG；返回按文件名 stem 排序的路径列表和模式"""
    raws = sorted(f for f in dir_path.iterdir() if f.suffix.lower() in RAW_EXTS)
    if raws:
        return raws, 'raw'
    pngs = sorted(f for f in dir_path.iterdir() if f.suffix.lower() == '.png')
    if pngs:
        return pngs, 'bayer_png'
    return [], 'empty'


class VideoFrameSource:
    """视频文件帧源：灰度视频=Bayer 流；彩色视频=模拟 Bayer (demo)"""

    def __init__(self, path: str, bayer_pattern: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频: {path}")
        self.bayer_pattern = bayer_pattern
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 20.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        # 判断模式需要读第一帧
        ok, first = self.cap.read()
        if not ok:
            raise RuntimeError(f"视频读取失败: {path}")
        self.mode = 'bayer' if first.ndim == 2 else 'demo'
        self._first = first

    def __iter__(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self

    def __next__(self):
        if self._first is not None:
            frame, self._first = self._first, None
        else:
            ok, frame = self.cap.read()
            if not ok:
                raise StopIteration
        return self.to_bayer(frame)

    def to_bayer(self, frame: np.ndarray) -> np.ndarray:
        if self.mode == 'bayer':
            return frame
        return simulate_bayer_raw(frame, self.bayer_pattern)

    def close(self):
        self.cap.release()


def frame_pairs(args):
    """统一帧对迭代器：yield (left_bayer_or_path, right_bayer_or_path, frame_id, timestamp, mode)"""
    left_path = Path(args.left)
    right_path = Path(args.right)

    if left_path.is_dir() and right_path.is_dir():
        left_files, lmode = load_dir_frames(left_path)
        right_files, rmode = load_dir_frames(right_path)
        if lmode != rmode:
            print(f"  ! 左右目录文件类型不一致 (左: {lmode}, 右: {rmode})，按各自类型处理")
        left_map = {f.stem: f for f in left_files}
        right_map = {f.stem: f for f in right_files}
        stems = sorted(set(left_map) & set(right_map))
        if not stems:
            raise RuntimeError(f"左右目录没有文件名匹配的帧对: {left_path} / {right_path}")
        print(f"目录模式: {lmode}, 匹配 {len(stems)} 对帧")
        for i, stem in enumerate(stems):
            if args.max_frames and i >= args.max_frames:
                break
            yield left_map[stem], right_map[stem], i, i / args.fps, lmode

    elif left_path.is_file() and right_path.is_file():
        ls, rs = VideoFrameSource(str(left_path), args.bayer_pattern), VideoFrameSource(str(right_path), args.bayer_pattern)
        print(f"视频模式: 左={ls.mode}({ls.total or '?'}帧 @{ls.fps:.1f}fps), 右={rs.mode}({rs.total or '?'}帧)")
        fps = ls.fps
        for i, (lb, rb) in enumerate(zip(ls, rs)):
            if args.max_frames and i >= args.max_frames:
                break
            yield lb, rb, i, i / fps, 'video'
        ls.close()
        rs.close()

    else:
        raise RuntimeError("左右路径必须同时是目录或同时是视频文件")


def process_bayer(pipeline: ISPPipeline, bayer: np.ndarray, args, pattern: str) -> np.ndarray:
    wl = auto_white_level(bayer, args)
    # uint8 数据（模拟/8bit流）黑电平为 0；uint16 沿用管线配置（典型 14bit RAW 的 512）
    bl = 0 if bayer.dtype == np.uint8 else None
    return pipeline.process_bayer_data(bayer, white_level=wl, bayer_pattern=pattern, black_level=bl)


def load_detector(model_path: str):
    """延迟加载 YOLO 模型（仅在 --detect 时导入 ultralytics，避免硬依赖）"""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("未安装 ultralytics，请先执行: pip install ultralytics")
    print(f"  检测模型: {model_path}")
    return YOLO(model_path)


def draw_detections(img: np.ndarray, detector, conf: float) -> np.ndarray:
    """在图像上绘制检测框，返回带框图像"""
    results = detector.predict(img, conf=conf, verbose=False)
    return results[0].plot() if results else img


def process_pair(pipeline, left_in, right_in, args, mode):
    """处理一对输入 → (left_srgb, right_srgb)。mode: raw / bayer_png / video"""
    if mode == 'raw':
        left_srgb, _ = pipeline.process_file(str(left_in))
        right_srgb, _ = pipeline.process_file(str(right_in))
    elif mode == 'bayer_png':
        lb = cv2.imread(str(left_in), cv2.IMREAD_UNCHANGED)
        rb = cv2.imread(str(right_in), cv2.IMREAD_UNCHANGED)
        left_srgb = process_bayer(pipeline, lb, args, args.bayer_pattern)
        right_srgb = process_bayer(pipeline, rb, args, args.bayer_pattern)
    else:  # video 模式，输入已经是 Bayer 数组
        left_srgb = process_bayer(pipeline, left_in, args, args.bayer_pattern)
        right_srgb = process_bayer(pipeline, right_in, args, args.bayer_pattern)
    return left_srgb, right_srgb


def main():
    args = parse_args()
    config_dir = project_root / 'configs'
    loader = ConfigLoader(str(config_dir))

    print("初始化 ISP 管线...")
    pipeline = ISPPipeline(loader.load_pipeline_config(args.config))
    checker = StereoConsistencyChecker() if args.check_consistency else None
    detector = load_detector(args.detect) if args.detect else None

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_frames:
        (out_dir / 'frames').mkdir(exist_ok=True)

    rectifier = None
    writers = {}
    summary = {'frames_processed': 0, 'mode': None, 'rectification': None,
               'consistency_failures': 0, 'detect_model': args.detect}
    verified_first = False

    for left_in, right_in, frame_id, ts, mode in frame_pairs(args):
        summary['mode'] = mode

        left_srgb, right_srgb = process_pair(pipeline, left_in, right_in, args, mode)
        h, w = left_srgb.shape[:2]

        # 校正器延迟构建：等第一帧拿到真实尺寸后构建，映射表只算一次
        if rectifier is None and not args.no_rectify:
            rectifier = build_rectifier(loader, args, (w, h))

        if rectifier is not None:
            left_srgb, right_srgb = rectifier.rectify(left_srgb, right_srgb)
            if not verified_first:
                res = rectifier.verify_rectification(left_srgb, right_srgb)
                print(f"  校正验证(首帧): 匹配{res['num_matches']}对, "
                      f"平均垂直误差={res['mean_vertical_error_px']:.3f}px, "
                      f"通过0.5px阈值={res['pass_THRESHOLD_0_5px']}")
                summary['rectification'] = res
                verified_first = True

        if checker is not None:
            c = checker.check(left_srgb, right_srgb)
            if not c['overall_pass']:
                summary['consistency_failures'] += 1
                print(f"  ! 第{frame_id}帧一致性未通过: 亮度差={c['brightness_diff']:.3f} 颜色差={c['color_diff']:.3f}")

        # 目标检测（可选）：在最终图像上画检测框
        if detector is not None:
            if args.detect_side in ('left', 'both'):
                left_srgb = draw_detections(left_srgb, detector, args.detect_conf)
            if args.detect_side in ('right', 'both'):
                right_srgb = draw_detections(right_srgb, detector, args.detect_conf)

        # 输出视频
        out_h, out_w = left_srgb.shape[:2]
        if args.layout in ('side_by_side', 'both'):
            key = 'side_by_side.mp4'
            if key not in writers:
                writers[key] = cv2.VideoWriter(str(out_dir / key),
                                               cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (out_w * 2, out_h))
            vis = np.hstack([left_srgb, right_srgb])
            cv2.putText(vis, f"L frame {frame_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis, f"R frame {frame_id}", (out_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            writers[key].write(vis)
        if args.layout in ('separate', 'both'):
            for side, img in (('left', left_srgb), ('right', right_srgb)):
                key = f'{side}.mp4'
                if key not in writers:
                    writers[key] = cv2.VideoWriter(str(out_dir / key),
                                                   cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (out_w, out_h))
                writers[key].write(img)

        if args.save_frames:
            cv2.imwrite(str(out_dir / 'frames' / f'{frame_id:06d}_left.png'), left_srgb)
            cv2.imwrite(str(out_dir / 'frames' / f'{frame_id:06d}_right.png'), right_srgb)

        summary['frames_processed'] += 1
        if frame_id % 25 == 0:
            print(f"  已处理 {summary['frames_processed']} 帧")

    for w_ in writers.values():
        w_.release()

    with open(out_dir / 'process_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n完成: 处理 {summary['frames_processed']} 帧, 输出目录: {out_dir.resolve()}")
    print(f"视频文件: {', '.join(writers.keys())}")


if __name__ == '__main__':
    main()
