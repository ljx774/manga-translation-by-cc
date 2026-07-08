"""OCR 引擎抽象基类"""

from abc import ABC, abstractmethod

from manga_translator.types import TextBlock


class BaseOCREngine(ABC):
    """OCR 引擎基类"""

    @abstractmethod
    def detect_and_recognize(self, image_path: str) -> list[TextBlock]:
        """
        检测图片中的文字并识别。

        Args:
            image_path: 图片文件路径

        Returns:
            TextBlock 列表，包含位置、文字和置信度
        """
        ...