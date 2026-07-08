"""文字渲染引擎 — 将翻译后的文字渲染到图片上"""

import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from manga_translator.rendering.base import BaseRenderer
from manga_translator.types import TranslatedBlock

logger = logging.getLogger(__name__)

# 常见系统 CJK 字体路径
_CJK_FONT_CANDIDATES = [
    # Noto Sans CJK (Linux)
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    # WenQuanYi (Linux)
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/wenquanyi/wqy-zenhei/wqy-zenhei.ttc",
    # DroidSansFallback (Linux)
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    # Windows
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/yugothib.ttc",
]


def find_cjk_font() -> str | None:
    """查找系统中可用的 CJK 字体"""
    for path in _CJK_FONT_CANDIDATES:
        if os.path.exists(path):
            return path

    # 尝试 fc-list 命令查找
    import subprocess
    try:
        result = subprocess.run(
            ["fc-list", ":lang=ja", "file"],
            capture_output=True, text=True, timeout=5,
        )
        if result.stdout:
            # 取第一个字体路径
            first_line = result.stdout.strip().split("\n")[0]
            font_path = first_line.split(":")[0].strip()
            if os.path.exists(font_path):
                return font_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


class TextRenderer(BaseRenderer):
    """文字渲染引擎

    将翻译后的文字渲染到图片上，自动计算字体大小和换行。
    """

    def __init__(
        self,
        font_path: str | None = None,
        font_size: int | None = None,
        font_color: tuple[int, int, int] = (0, 0, 0),
        min_font_size: int = 10,
        max_font_size: int = 32,
        text_padding: int = 6,
    ):
        """
        Args:
            font_path: 字体文件路径，None 则自动查找
            font_size: 固定字体大小，None 则自动计算
            font_color: 字体颜色 RGB
            min_font_size: 自动计算的最小字体大小
            max_font_size: 自动计算的最大字体大小
            text_padding: 文字与边界的内边距
        """
        self.font_path = font_path or find_cjk_font()
        self.font_size = font_size
        self.font_color = font_color
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size
        self.text_padding = text_padding

        if not self.font_path:
            logger.warning(
                "未找到 CJK 字体！中文/日文/韩文可能无法正常渲染。"
                "请安装 fonts-noto-cjk 或指定字体路径。"
            )

    def render(
        self, image: np.ndarray, blocks: list[TranslatedBlock]
    ) -> np.ndarray:
        """
        将翻译后的文字渲染到图片上。

        Args:
            image: 已擦除的图片 (H, W, C) BGR 格式
            blocks: 翻译后的文本块

        Returns:
            渲染后的图片
        """
        # 转换为 PIL Image (BGR → RGB)
        image_rgb = cv2_to_pil(image)
        draw = ImageDraw.Draw(image_rgb)

        for block in blocks:
            x1, y1, x2, y2 = block.bbox
            text = block.translated

            if not text.strip():
                continue

            # 计算可用区域
            avail_w = (x2 - x1) - 2 * self.text_padding
            avail_h = (y2 - y1) - 2 * self.text_padding

            if avail_w <= 0 or avail_h <= 0:
                continue

            # 确定字体大小
            if self.font_size:
                font_size = self.font_size
            else:
                font_size = self._calculate_font_size(
                    draw, text, avail_w, avail_h
                )

            # 加载字体
            font = self._load_font(font_size)

            # 换行处理
            lines = self._wrap_text(draw, text, font, avail_w)

            # 计算文字总高度
            total_text_height = sum(
                self._get_text_height(draw, line, font) for line in lines
            )
            line_spacing = 2
            total_text_height += line_spacing * (len(lines) - 1)

            # 垂直居中起始位置
            text_y = y1 + self.text_padding + (avail_h - total_text_height) / 2

            # 逐行绘制
            for line in lines:
                text_width = draw.textlength(line, font=font)
                text_height = self._get_text_height(draw, line, font)

                # 水平居中
                text_x = x1 + self.text_padding + (avail_w - text_width) / 2

                # 绘制文字（PIL 的 text 方法需要整数坐标）
                draw.text(
                    (int(text_x), int(text_y)),
                    line,
                    font=font,
                    fill=self.font_color,
                )

                text_y += text_height + line_spacing

            logger.debug("渲染文字: '%s' @ (%d,%d)-(%d,%d) font=%d",
                         text[:20], x1, y1, x2, y2, font_size)

        # 转回 numpy BGR
        return pil_to_cv2(image_rgb)

    def _calculate_font_size(
        self,
        draw: ImageDraw.Draw,
        text: str,
        max_width: int,
        max_height: int,
    ) -> int:
        """二分查找合适的字体大小"""
        lo, hi = self.min_font_size, self.max_font_size
        best_size = self.min_font_size

        while lo <= hi:
            mid = (lo + hi) // 2
            font = self._load_font(mid)
            lines = self._wrap_text(draw, text, font, max_width)

            total_h = sum(
                self._get_text_height(draw, line, font) for line in lines
            )
            total_h += 2 * (len(lines) - 1)  # line spacing

            if total_h <= max_height:
                best_size = mid
                lo = mid + 1
            else:
                hi = mid - 1

        return best_size

    def _wrap_text(
        self,
        draw: ImageDraw.Draw,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_width: int,
    ) -> list[str]:
        """将文本按可用宽度换行"""
        if not text:
            return []

        lines = []
        current_line = ""

        for char in text:
            test_line = current_line + char
            if draw.textlength(test_line, font=font) <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = char

        if current_line:
            lines.append(current_line)

        return lines or [text]

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont:
        """加载指定大小的字体"""
        if self.font_path:
            try:
                return ImageFont.truetype(self.font_path, size)
            except OSError:
                logger.warning("无法加载字体 %s，使用默认字体", self.font_path)
        return ImageFont.load_default()

    @staticmethod
    def _get_text_height(
        draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont
    ) -> int:
        """获取文字高度"""
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[3] - bbox[1]


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """OpenCV BGR 转 PIL RGB"""
    import cv2
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """PIL RGB 转 OpenCV BGR"""
    import cv2
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)