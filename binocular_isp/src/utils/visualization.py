"""
工具模块：可视化调试
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple


class DebugVisualizer:
    """调试可视化工具"""

    @staticmethod
    def show_image(img: np.ndarray, title: str = "Image", wait_key: bool = True):
        """显示图像（OpenCV 窗口）"""
        cv2.imshow(title, img)
        if wait_key:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    @staticmethod
    def save_debug_image(img: np.ndarray, path: str):
        """保存调试图像"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img)

    @staticmethod
    def plot_histogram(img: np.ndarray, title: str = "Histogram", save_path: Optional[str] = None):
        """绘制直方图"""
        colors = ('r', 'g', 'b')
        plt.figure(figsize=(10, 4))
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
        plt.title(title)
        plt.xlim([0, 256])
        if save_path:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def plot_side_by_side(left: np.ndarray, right: np.ndarray, 
                          titles: Tuple[str, str] = ("Left", "Right"),
                          save_path: Optional[str] = None):
        """并排显示左右图像"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # BGR → RGB for matplotlib
        axes[0].imshow(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
        axes[0].set_title(titles[0])
        axes[0].axis('off')
        axes[1].imshow(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))
        axes[1].set_title(titles[1])
        axes[1].axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def plot_raw_bayer(bayer_data: np.ndarray, title: str = "Bayer RAW", 
                       save_path: Optional[str] = None):
        """显示 RAW Bayer 数据"""
        plt.figure(figsize=(8, 6))
        plt.imshow(bayer_data, cmap='gray', vmin=0, vmax=np.percentile(bayer_data, 99))
        plt.title(title)
        plt.colorbar(label='Pixel Value')
        if save_path:
            plt.savefig(save_path)
        plt.show()
