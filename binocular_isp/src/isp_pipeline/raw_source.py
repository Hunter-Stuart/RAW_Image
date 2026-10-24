"""
统一 RAW 帧源抽象层

让"读 RAW 文件（rawpy）"与"读 FLIR 相机流（Spinnaker/PySpin）"走同一接口，
管线只依赖 RawFrame 数据，不关心数据来源。FLIR 相机到货后只需在
SpinnakerStereoSource 里填充/调试采集细节，管线代码零改动。

用法示例:
    from isp_pipeline.raw_source import RawpyFileSource

    with RawpyFileSource('data/input/aodraw') as src:
        for frame in src:
            rgb = pipeline.process(
                frame.bayer,
                metadata={'black_level': frame.black_level,
                          'white_level': frame.white_level},
                bayer_pattern=frame.bayer_pattern,
            )
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

RAW_EXTS = ('.arw', '.dng', '.cr2', '.nef', '.raw')

# FLIR 像素格式 → 本项目 Bayer 排列命名
FLIR_BAYER_TO_PATTERN = {
    'BayerRG8': 'RGGB', 'BayerRG12': 'RGGB', 'BayerRG16': 'RGGB',
    'BayerGB8': 'GBRG', 'BayerGB12': 'GBRG', 'BayerGB16': 'GBRG',
    'BayerGR8': 'GRBG', 'BayerGR12': 'GRBG', 'BayerGR16': 'GRBG',
    'BayerBG8': 'BGGR', 'BayerBG12': 'BGGR', 'BayerBG16': 'BGGR',
}


def bayer_pattern_from_flir(pixel_format: str) -> str:
    """FLIR 像素格式名 → 本项目 Bayer 模式名"""
    return FLIR_BAYER_TO_PATTERN.get(pixel_format, 'RGGB')


def bit_depth_from_flir(pixel_format: str) -> int:
    """FLIR 像素格式名 → 位深"""
    if '8' in pixel_format:
        return 8
    if '12' in pixel_format:
        return 12
    return 16


@dataclass
class RawFrame:
    """统一的原始帧数据结构：无论来自文件还是相机"""
    bayer: np.ndarray                        # HxW Bayer 阵列
    bayer_pattern: str = 'RGGB'
    bit_depth: int = 12
    black_level: int = 0
    white_level: int = 4095
    exposure_us: float = 0.0
    gain_db: float = 0.0
    timestamp_ns: int = 0                    # 设备时间戳，双目同步验证用
    frame_id: int = 0
    source: str = ''

    def to_metadata(self) -> dict:
        """转换为管线 process() 需要的 metadata 字典"""
        return {
            'black_level': self.black_level,
            'white_level': self.white_level,
        }


class RawSource(abc.ABC):
    """帧源抽象基类：迭代器接口 + 资源管理"""

    @abc.abstractmethod
    def __next__(self) -> RawFrame:
        ...

    def close(self) -> None:
        pass

    def __iter__(self) -> 'RawSource':
        return self

    def __enter__(self) -> 'RawSource':
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class RawpyFileSource(RawSource):
    """从 RAW 文件读取（rawpy），支持单个文件或目录（目录内按文件名排序）"""

    def __init__(self, path: str, bayer_pattern: str = 'RGGB'):
        self.path = Path(path)
        self.bayer_pattern = bayer_pattern
        if self.path.is_dir():
            self.files = sorted(f for f in self.path.iterdir()
                                if f.suffix.lower() in RAW_EXTS)
        elif self.path.is_file():
            self.files = [self.path]
        else:
            raise FileNotFoundError(f'路径不存在: {path}')
        if not self.files:
            raise FileNotFoundError(f'未找到 RAW 文件: {path}')
        self._idx = 0
        self._raw = None

    def __next__(self) -> RawFrame:
        if self._idx >= len(self.files):
            raise StopIteration
        f = self.files[self._idx]
        self._idx += 1

        import rawpy
        self.close()  # 释放上一帧
        self._raw = rawpy.imread(str(f))

        bayer = np.array(self._raw.raw_image_visible, dtype=np.uint16)
        black = self._raw.black_level_per_channel
        black = int(np.mean(np.asarray(black))) if black is not None else 0
        white = int(self._raw.white_level or 4095)

        color_desc = getattr(self._raw, 'color_desc', None)
        desc = color_desc.decode('utf-8') if isinstance(color_desc, bytes) \
            else (str(color_desc) if color_desc else self.bayer_pattern)
        from .raw_reader import normalize_bayer_pattern
        pattern = normalize_bayer_pattern(desc)

        return RawFrame(
            bayer=bayer,
            bayer_pattern=pattern,
            bit_depth=16 if white > 4095 else 12,
            black_level=black,
            white_level=white,
            frame_id=self._idx - 1,
            source=str(f),
        )

    def close(self) -> None:
        if self._raw is not None:
            self._raw.close()
            self._raw = None


class SpinnakerStereoSource(RawSource):
    """
    FLIR 双目相机同步采集（Spinnaker/PySpin）

    骨架已按官方示例（Acquisition/StereoGPIO/Trigger）写好，采集细节需在
    相机到货后按实测调试（见代码中 TODO）。设计要点:
      - 主从 GPIO 同步: 主相机 Line1 输出 ExposureActive → 从相机 Line0 输入触发
      - 关闭相机片上处理（色彩/伽马），只取裸 Bayer
      - 12-bit 时 GetNDArray 返回 uint16 容器，数据是否左移需实测确认
    """

    def __init__(self,
                 serial_left: Optional[str] = None,
                 serial_right: Optional[str] = None,
                 pixel_format: str = 'BayerRG12',
                 exposure_us: float = 10000.0,
                 gain_db: float = 0.0):
        import PySpin
        self.PySpin = PySpin
        self.pixel_format = pixel_format
        self.exposure_us = exposure_us
        self.gain_db = gain_db
        self.frame_id = 0

        self.system = PySpin.System.GetInstance()
        cam_list = self.system.GetCameras()
        n = cam_list.GetSize()
        if n < 2:
            cam_list.Clear()
            self.system.ReleaseInstance()
            raise RuntimeError(
                f'未检测到双目相机（当前 {n} 台）。请检查: '
                '两相机已上电并连接 USB3、串口被系统识别')

        # 按序列号选择左右相机；未指定则按枚举顺序取前两台
        cams = [cam_list.GetByIndex(i) for i in range(n)]
        cam_list.Clear()
        if serial_left:
            cams = sorted(cams, key=lambda c: c.DeviceSerialNumber() != serial_left)
        self.cam_left = cams[0]
        self.cam_right = cams[1]
        for cam in self.cam_left, self.cam_right:
            cam.Init()

        self._configure(self.cam_left, is_master=True)
        self._configure(self.cam_right, is_master=False)
        self._configure_sync()

    def _configure(self, cam, is_master: bool) -> None:
        """配置像素格式与曝光参数，关闭片上处理"""
        # 像素格式（裸 Bayer）
        cam.PixelFormat.SetIntValue(
            getattr(self.PySpin, f'PixelFormat_{self.pixel_format}'))
        # 关闭自动曝光/自动增益，手动锁定（双目一致性要求）
        cam.ExposureAuto.SetIntValue(self.PySpin.ExposureAuto_Off)
        cam.GainAuto.SetIntValue(self.PySpin.GainAuto_Off)
        cam.ExposureTime.SetValue(self.exposure_us)
        cam.Gain.SetValue(self.gain_db)
        # 关闭相机片上色彩处理，只取裸 Bayer
        try:
            cam.ColorTransformationEnable.SetValue(False)
        except Exception:
            pass  # 部分机型无此节点

        if is_master:
            # 主相机: Line1 输出曝光有效信号，供从相机触发
            cam.LineSelector.SetIntValue(self.PySpin.LineSelector_Line1)
            cam.LineMode.SetIntValue(self.PySpin.LineMode_Output)
            cam.LineSource.SetIntValue(self.PySpin.LineSource_ExposureActive)
            print(f'  主相机 {cam.DeviceSerialNumber()}: 触发输出 Line1=ExposureActive')
        else:
            # 从相机: Line0 输入触发
            cam.TriggerSelector.SetIntValue(self.PySpin.TriggerSelector_FrameStart)
            cam.TriggerMode.SetIntValue(self.PySpin.TriggerMode_On)
            cam.TriggerSource.SetIntValue(self.PySpin.TriggerSource_Line0)
            cam.TriggerActivation.SetIntValue(self.PySpin.TriggerActivation_RisingEdge)
            print(f'  从相机 {cam.DeviceSerialNumber()}: 触发输入 Line0=FrameStart')

    def _configure_sync(self) -> None:
        """双机连续采集（主相机自由运行，从相机由 GPIO 触发）"""
        for cam in self.cam_left, self.cam_right:
            cam.AcquisitionMode.SetIntValue(self.PySpin.AcquisitionMode_Continuous)
            cam.BeginAcquisition()
        print('  双目采集已启动 (GPIO 主从同步)')

    def __next__(self) -> Tuple[RawFrame, RawFrame]:
        """返回 (左帧, 右帧)。从相机等待触发信号，主相机按帧率出帧"""
        # TODO(设备到货后实测): 两路 GetNextImage 的同步策略 —
        # 从相机由 GPIO 触发，两路帧率一致时可顺序取图；
        # 帧率不一致时需按 FrameID/时间戳配对，丢掉失配帧
        img_left = self.cam_left.GetNextImage(self.PySpin.EVENT_TIMEOUT_INFINITE)
        img_right = self.cam_right.GetNextImage(self.PySpin.EVENT_TIMEOUT_INFINITE)

        left = self._to_frame(img_left, 'left')
        right = self._to_frame(img_right, 'right')
        self.frame_id += 1
        left.frame_id = right.frame_id = self.frame_id

        img_left.Release()
        img_right.Release()
        return left, right

    def _to_frame(self, img, side: str) -> RawFrame:
        pattern = bayer_pattern_from_flir(self.pixel_format)
        depth = bit_depth_from_flir(self.pixel_format)
        # TODO(设备到货后实测): BayerRG12 的 GetNDArray 数据可能左移（低12位为0），
        # 打印 img.max() 确认后再定 white_level 是否要右移
        black = int(self.cam_left.BlackLevel.GetValue()) if side == 'left' \
            else int(self.cam_right.BlackLevel.GetValue())
        return RawFrame(
            bayer=img.GetNDArray(),
            bayer_pattern=pattern,
            bit_depth=depth,
            black_level=black,
            white_level=(1 << depth) - 1,
            exposure_us=float(img.GetExposureTime()),
            timestamp_ns=int(img.GetTimeStamp()),
            source=side,
        )

    def close(self) -> None:
        for cam in self.cam_left, self.cam_right:
            try:
                cam.EndAcquisition()
                cam.DeInit()
            except Exception:
                pass
        del self.cam_left, self.cam_right
        self.system.ReleaseInstance()
