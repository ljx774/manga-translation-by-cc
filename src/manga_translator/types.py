"""共享数据类型"""

from dataclasses import dataclass


@dataclass
class TextBlock:
    """OCR 检测到的文本块"""
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) 左上角+右下角
    text: str                         # 识别出的文字
    confidence: float                 # 置信度 [0, 1]


@dataclass
class TranslatedBlock:
    """翻译后的文本块"""
    bbox: tuple[int, int, int, int]
    original: str
    translated: str