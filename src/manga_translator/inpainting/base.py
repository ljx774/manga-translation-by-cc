"""擦除/修复引擎抽象基类"""

from abc import ABC, abstractmethod

import numpy as np

from manga_translator.types import TranslatedBlock


class BaseInpainter(ABC):
    """擦除引擎基类"""

    @abstractmethod
    def inpaint(self, image: np.ndarray, blocks: list[TranslatedBlock]) -> np.ndarray:
        """
        擦除图片中的原文区域。

        Args:
            image: 原始图片 (H, W, C) numpy 数组
            blocks: 翻译后的文本块列表（含 bbox 信息）

        Returns:
            擦除后的图片 numpy 数组
        """
        ...