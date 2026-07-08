"""翻译引擎抽象基类"""

from abc import ABC, abstractmethod

from manga_translator.types import TextBlock, TranslatedBlock


class BaseTranslator(ABC):
    """翻译引擎基类"""

    @abstractmethod
    def translate(self, blocks: list[TextBlock], source_lang: str, target_lang: str) -> list[TranslatedBlock]:
        """
        翻译文本块列表。

        Args:
            blocks: 待翻译的文本块列表
            source_lang: 源语言代码 (ja, zh, en, ko 等)
            target_lang: 目标语言代码

        Returns:
            TranslatedBlock 列表
        """
        ...