"""EasyOCR 引擎实现"""

import logging

import numpy as np

from manga_translator.ocr.base import BaseOCREngine
from manga_translator.types import TextBlock

logger = logging.getLogger(__name__)


class EasyOCREngine(BaseOCREngine):
    """基于 EasyOCR 的文字检测与识别引擎"""

    def __init__(
        self,
        source_lang: str = "ja",
        gpu: bool = True,
        min_confidence: float = 0.5,
    ):
        """
        Args:
            source_lang: 源语言代码，EasyOCR 支持 'ja', 'zh', 'en', 'ko' 等
            gpu: 是否使用 GPU 加速
            min_confidence: 最低置信度阈值
        """
        self.source_lang = source_lang
        self.gpu = gpu
        self.min_confidence = min_confidence
        self._reader = None

    @property
    def reader(self):
        """延迟初始化 EasyOCR Reader"""
        if self._reader is None:
            import easyocr

            # EasyOCR 语言代码映射
            lang_list = self._build_lang_list(self.source_lang)
            logger.info(
                "初始化 EasyOCR: langs=%s, gpu=%s", lang_list, self.gpu
            )
            self._reader = easyocr.Reader(lang_list, gpu=self.gpu)
        return self._reader

    @staticmethod
    def _build_lang_list(lang: str) -> list[str]:
        """构建 EasyOCR 的语言列表，包含英文作为辅助"""
        # EasyOCR 支持的语言代码
        if lang == "zh":
            # 中文需要同时指定简体和繁体
            return ["ch_sim", "en"]
        elif lang == "ja":
            return ["ja", "en"]
        elif lang == "ko":
            return ["ko", "en"]
        else:
            return [lang]

    def detect_and_recognize(self, image_path: str) -> list[TextBlock]:
        """
        检测并识别图片中的文字。

        Args:
            image_path: 图片路径

        Returns:
            TextBlock 列表
        """
        logger.info("开始 OCR 处理: %s", image_path)

        results = self.reader.readtext(image_path)

        blocks = []
        for bbox, text, confidence in results:
            if confidence < self.min_confidence:
                logger.debug("跳过低置信度文本: '%s' (%.2f)", text, confidence)
                continue

            # EasyOCR 返回的 bbox 是 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] 四点格式
            # 转换为 (x1, y1, x2, y2) 矩形格式
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            rect = (
                int(min(x_coords)),
                int(min(y_coords)),
                int(max(x_coords)),
                int(max(y_coords)),
            )

            block = TextBlock(
                bbox=rect,
                text=text.strip(),
                confidence=float(confidence),
            )
            blocks.append(block)
            logger.debug("检测到文本: '%s' @ %s (%.2f)", text, rect, confidence)

        logger.info("OCR 完成，共检测到 %d 个文本块", len(blocks))
        return blocks