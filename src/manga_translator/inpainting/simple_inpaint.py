"""简单擦除引擎 — 基于背景色填充"""

import logging

import cv2
import numpy as np

from manga_translator.inpainting.base import BaseInpainter
from manga_translator.types import TranslatedBlock

logger = logging.getLogger(__name__)


class SimpleInpainter(BaseInpainter):
    """简单擦除引擎

    通过检测文本区域周围的背景色，用该颜色填充文本区域来擦除原文。
    适用于漫画气泡（通常是白色或浅色背景）的场景。
    """

    def __init__(self, padding: int = 4, blur_radius: int = 3):
        """
        Args:
            padding: 文本区域扩展像素（确保完全覆盖原文）
            blur_radius: 边缘模糊半径（使填充区域边缘更自然）
        """
        self.padding = padding
        self.blur_radius = blur_radius

    def inpaint(
        self, image: np.ndarray, blocks: list[TranslatedBlock]
    ) -> np.ndarray:
        """
        擦除图片中的原文。

        Args:
            image: 原始图片 (H, W, C) BGR 格式
            blocks: 翻译后的文本块

        Returns:
            擦除后的图片
        """
        result = image.copy()
        h, w = result.shape[:2]

        for block in blocks:
            x1, y1, x2, y2 = block.bbox

            # 扩展区域
            x1 = max(0, x1 - self.padding)
            y1 = max(0, y1 - self.padding)
            x2 = min(w, x2 + self.padding)
            y2 = min(h, y2 + self.padding)

            if x1 >= x2 or y1 >= y2:
                continue

            # 检测背景色：取文本区域外围一圈像素的中位数颜色
            bg_color = self._detect_background_color(result, x1, y1, x2, y2)

            # 创建填充用的 mask
            mask = self._create_text_mask(result, x1, y1, x2, y2)

            # 用背景色填充
            result[y1:y2, x1:x2] = np.where(
                mask[..., np.newaxis],
                bg_color,
                result[y1:y2, x1:x2],
            )

            # 边缘模糊
            if self.blur_radius > 0:
                roi = result[y1:y2, x1:x2]
                blurred = cv2.GaussianBlur(roi, (self.blur_radius, self.blur_radius), 0)
                # 只在填充区域应用模糊
                result[y1:y2, x1:x2] = np.where(
                    mask[..., np.newaxis],
                    blurred,
                    result[y1:y2, x1:x2],
                )

            logger.debug(
                "擦除文本区域: (%d,%d)-(%d,%d) 背景色=%s",
                x1, y1, x2, y2, bg_color,
            )

        return result

    def _detect_background_color(
        self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> np.ndarray:
        """检测文本区域周围的背景色

        取区域外围一圈像素的中位数颜色，作为气泡的背景色。
        """
        h, w = image.shape[:2]

        # 收集区域外围像素
        border_pixels = []

        # 上边
        if y1 > 0:
            border_pixels.append(image[y1 - 1, max(0, x1):min(w, x2)])
        # 下边
        if y2 < h:
            border_pixels.append(image[y2, max(0, x1):min(w, x2)])
        # 左边
        if x1 > 0:
            border_pixels.append(image[max(0, y1):min(h, y2), x1 - 1])
        # 右边
        if x2 < w:
            border_pixels.append(image[max(0, y1):min(h, y2), x2])

        if border_pixels:
            all_border = np.concatenate([p.reshape(-1, 3) for p in border_pixels])
            bg_color = np.median(all_border, axis=0).astype(np.uint8)
        else:
            # 默认白色
            bg_color = np.array([255, 255, 255], dtype=np.uint8)

        return bg_color

    def _create_text_mask(
        self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> np.ndarray:
        """创建文字区域的二值 mask

        通过检测与背景色差异较大的像素来确定文字位置。
        """
        roi = image[y1:y2, x1:x2]
        bg_color = self._detect_background_color(image, x1, y1, x2, y2)

        # 计算每个像素与背景色的差异
        diff = np.abs(roi.astype(np.float32) - bg_color.astype(np.float32))
        diff = np.mean(diff, axis=2)

        # 阈值化：差异较大的像素被认为是文字
        threshold = 40
        mask = diff > threshold

        # 形态学膨胀，确保覆盖文字边缘
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)

        return mask.astype(bool)