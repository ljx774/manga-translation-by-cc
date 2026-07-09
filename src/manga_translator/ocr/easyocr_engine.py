"""EasyOCR 引擎实现"""

import logging

import cv2
import numpy as np

from manga_translator.ocr.base import BaseOCREngine
from manga_translator.types import TextBlock

logger = logging.getLogger(__name__)


class EasyOCREngine(BaseOCREngine):
    """基于 EasyOCR 的文字检测与识别引擎

    支持横排和竖排文字（日文漫画常见从右到左竖排）。
    竖排处理：旋转图像检测文字列位置 → 映射回原图坐标 → 在原图上识别。
    """

    def __init__(
        self,
        source_lang: str = "ja",
        gpu: bool = True,
        min_confidence: float = 0.5,
    ):
        self.source_lang = source_lang
        self.gpu = gpu
        self.min_confidence = min_confidence
        self._reader = None

    @property
    def reader(self):
        """延迟初始化 EasyOCR Reader"""
        if self._reader is None:
            import easyocr

            lang_list = self._build_lang_list(self.source_lang)
            logger.info(
                "初始化 EasyOCR: langs=%s, gpu=%s", lang_list, self.gpu
            )
            self._reader = easyocr.Reader(lang_list, gpu=self.gpu)
        return self._reader

    @staticmethod
    def _build_lang_list(lang: str) -> list[str]:
        """构建 EasyOCR 的语言列表"""
        if lang == "zh":
            return ["ch_sim", "en"]
        elif lang == "ja":
            return ["ja", "en"]
        elif lang == "ko":
            return ["ko", "en"]
        else:
            return [lang]

    def detect_and_recognize(self, image_path: str) -> list[TextBlock]:
        """检测并识别图片中的文字（支持横排和竖排）。"""
        logger.info("开始 OCR 处理: %s", image_path)

        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"无法读取图片: {image_path}")

        h, w = image.shape[:2]
        blocks: list[TextBlock] = []

        # 1. 原图 OCR（横排文字：检测 + 识别）
        logger.info("OCR 横排文字...")
        horizontal_results = self.reader.readtext(image)
        blocks.extend(self._parse_results(horizontal_results))
        logger.info("  横排: %d 个文本块", len(horizontal_results))

        # 2. 竖排文字：旋转后检测 → 映射回原图 → 原图识别
        logger.info("OCR 竖排文字...")
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        # 在旋转图上检测文字位置（竖排文字在旋转后变成横排，检测器能识别）
        rotated_horizontal, rotated_free = self.reader.detect(rotated)

        # 合并两种检测结果，转换为多边形格式映射回原图
        # detect() 返回的 h_list 和 f_list 都是 [[...], [...]] 嵌套格式
        original_boxes = []

        # horizontal_list: [[cx, cy, w, h], ...] → 转为 4 点坐标 → 映射
        for batch in rotated_horizontal:
            for box in batch:
                cx, cy, rw, rh = box
                corners = [
                    (cx - rw / 2, cy - rh / 2),
                    (cx + rw / 2, cy - rh / 2),
                    (cx + rw / 2, cy + rh / 2),
                    (cx - rw / 2, cy + rh / 2),
                ]
                # 映射回原图: (rx, ry) → (h - ry, rx)
                mapped = [(h - ry, rx) for rx, ry in corners]
                original_boxes.append(mapped)

        # free_list: [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ...] → 映射
        for batch in rotated_free:
            for box in batch:
                mapped = [(h - p[1], p[0]) for p in box]
                original_boxes.append(mapped)

        if original_boxes:
            # 在原图上识别（文字是直立的，识别器能正确读取）
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            vertical_results = self.reader.recognize(
                gray,
                horizontal_list=[],
                free_list=original_boxes,
            )
            logger.info("  竖排: %d 个文本块", len(vertical_results))

            for bbox, text, confidence in vertical_results:
                if confidence < self.min_confidence:
                    continue
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                rect = (
                    int(min(x_coords)),
                    int(min(y_coords)),
                    int(max(x_coords)),
                    int(max(y_coords)),
                )
                rect = self._clamp_bbox(rect, w, h)
                blocks.append(TextBlock(
                    bbox=rect,
                    text=text.strip(),
                    confidence=float(confidence),
                ))
                logger.debug("竖排: '%s' @ %s (%.2f)", text, rect, confidence)
        else:
            logger.info("  竖排: 未检测到文字")

        # 3. 去重
        blocks = self._deduplicate_blocks(blocks)

        logger.info("OCR 完成，共检测到 %d 个文本块（去重后）", len(blocks))
        return blocks

    def _parse_results(self, results: list) -> list[TextBlock]:
        """将 EasyOCR 原始结果解析为 TextBlock 列表"""
        blocks = []
        for bbox, text, confidence in results:
            if confidence < self.min_confidence:
                continue
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            rect = (
                int(min(x_coords)),
                int(min(y_coords)),
                int(max(x_coords)),
                int(max(y_coords)),
            )
            blocks.append(TextBlock(
                bbox=rect,
                text=text.strip(),
                confidence=float(confidence),
            ))
        return blocks

    @staticmethod
    def _clamp_bbox(bbox: tuple, w: int, h: int) -> tuple:
        x1, y1, x2, y2 = bbox
        return (max(0, x1), max(0, y1), min(w, x2), min(h, y2))

    @staticmethod
    def _deduplicate_blocks(blocks: list[TextBlock], iou_threshold: float = 0.5) -> list[TextBlock]:
        """移除重叠度过高的重复文本块，保留置信度更高的"""
        if len(blocks) <= 1:
            return blocks

        sorted_blocks = sorted(blocks, key=lambda b: b.confidence, reverse=True)
        kept = []

        for block in sorted_blocks:
            is_duplicate = False
            for existing in kept:
                if EasyOCREngine._iou(block.bbox, existing.bbox) > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(block)

        return kept

    @staticmethod
    def _iou(bbox1: tuple, bbox2: tuple) -> float:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0