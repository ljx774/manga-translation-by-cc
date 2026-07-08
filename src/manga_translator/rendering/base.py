"""文字渲染引擎抽象基类"""

from abc import ABC, abstractmethod

import numpy as np

from manga_translator.types import TranslatedBlock


class BaseRenderer(ABC):
    """渲染引擎基类"""

    @abstractmethod
    def render(self, image: np.ndarray, blocks: list[TranslatedBlock]) -> np.ndarray:
        """
        将翻译后的文字渲染到图片上。

        Args:
            image: 已擦除原文的图片 (H, W, C) numpy 数组
            blocks: 翻译后的文本块列表

        Returns:
            渲染后的图片 numpy 数组
        """
        ...