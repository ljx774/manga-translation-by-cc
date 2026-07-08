"""漫画翻译流程编排"""

import logging
from pathlib import Path

import cv2
import numpy as np

from manga_translator.config import Config
from manga_translator.ocr.base import BaseOCREngine
from manga_translator.ocr.easyocr_engine import EasyOCREngine
from manga_translator.translation.base import BaseTranslator
from manga_translator.translation.claude_translator import ClaudeTranslator
from manga_translator.translation.openai_translator import OpenAITranslator
from manga_translator.inpainting.base import BaseInpainter
from manga_translator.inpainting.simple_inpaint import SimpleInpainter
from manga_translator.rendering.base import BaseRenderer
from manga_translator.rendering.text_renderer import TextRenderer
from manga_translator.types import TextBlock, TranslatedBlock

logger = logging.getLogger(__name__)


class MangaTranslator:
    """漫画翻译流水线

    串联 OCR → 翻译 → 擦除 → 渲染 四个步骤。
    """

    def __init__(
        self,
        config: Config | None = None,
        ocr_engine: BaseOCREngine | None = None,
        translator: BaseTranslator | None = None,
        inpainter: BaseInpainter | None = None,
        renderer: BaseRenderer | None = None,
    ):
        """
        Args:
            config: 全局配置，不传则使用默认配置
            ocr_engine: OCR 引擎，不传则根据配置自动创建
            translator: 翻译引擎，不传则根据配置自动创建
            inpainter: 擦除引擎，不传则根据配置自动创建
            renderer: 渲染引擎，不传则根据配置自动创建
        """
        self.config = config or Config()

        # 初始化各引擎
        self.ocr_engine = ocr_engine or self._create_ocr_engine()
        self.translator = translator or self._create_translator()
        self.inpainter = inpainter or self._create_inpainter()
        self.renderer = renderer or self._create_renderer()

    def _create_ocr_engine(self) -> BaseOCREngine:
        cfg = self.config.ocr_config
        engine = cfg.get("engine", "easyocr")
        if engine == "easyocr":
            return EasyOCREngine(
                source_lang=cfg.get("source_lang", "ja"),
                gpu=cfg.get("gpu", True),
                min_confidence=cfg.get("min_confidence", 0.5),
            )
        raise ValueError(f"不支持的 OCR 引擎: {engine}")

    def _create_translator(self) -> BaseTranslator:
        cfg = self.config.translation_config
        engine = cfg.get("engine", "openai")
        model = cfg.get("model", "gpt-4o")

        if engine == "openai":
            return OpenAITranslator(model=model)
        elif engine == "claude":
            return ClaudeTranslator(model=model)
        raise ValueError(f"不支持的翻译引擎: {engine}")

    def _create_inpainter(self) -> BaseInpainter:
        cfg = self.config.inpainting_config
        engine = cfg.get("engine", "simple")
        if engine == "simple":
            return SimpleInpainter(
                padding=cfg.get("padding", 4),
                blur_radius=cfg.get("blur_radius", 3),
            )
        raise ValueError(f"不支持的擦除引擎: {engine}")

    def _create_renderer(self) -> BaseRenderer:
        cfg = self.config.rendering_config
        font_color = tuple(cfg.get("font_color", [0, 0, 0]))
        return TextRenderer(
            font_path=cfg.get("font_path"),
            font_size=cfg.get("font_size"),
            font_color=font_color,
            min_font_size=cfg.get("min_font_size", 10),
            max_font_size=cfg.get("max_font_size", 32),
            text_padding=cfg.get("text_padding", 6),
        )

    def translate_image(
        self,
        image_path: str | Path,
        output_path: str | Path | None = None,
        source_lang: str | None = None,
        target_lang: str | None = None,
    ) -> Path:
        """
        翻译单张图片。

        Args:
            image_path: 输入图片路径
            output_path: 输出路径，None 则自动生成
            source_lang: 源语言（覆盖配置）
            target_lang: 目标语言（覆盖配置）

        Returns:
            输出文件路径
        """
        image_path = Path(image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"图片不存在: {image_path}")

        src_lang = source_lang or self.config.translation_config.get("source_lang", "ja")
        tgt_lang = target_lang or self.config.translation_config.get("target_lang", "zh")

        logger.info("=" * 50)
        logger.info("开始翻译: %s", image_path.name)
        logger.info("源语言: %s → 目标语言: %s", src_lang, tgt_lang)

        # 1. OCR
        logger.info("步骤 1/4: OCR 文字检测与识别...")
        text_blocks: list[TextBlock] = self.ocr_engine.detect_and_recognize(
            str(image_path)
        )
        logger.info("  检测到 %d 个文本块", len(text_blocks))

        if not text_blocks:
            logger.warning("未检测到任何文字，跳过翻译")
            if output_path:
                return Path(output_path)
            return image_path

        for i, block in enumerate(text_blocks):
            logger.info("  [%d] %s", i, block.text[:50])

        # 2. 翻译
        logger.info("步骤 2/4: 翻译...")
        translated_blocks: list[TranslatedBlock] = self.translator.translate(
            text_blocks, src_lang, tgt_lang
        )
        for i, block in enumerate(translated_blocks):
            logger.info("  [%d] %s → %s", i, block.original[:30], block.translated[:30])

        # 3. 擦除
        logger.info("步骤 3/4: 擦除原文...")
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"无法读取图片: {image_path}")

        inpainted = self.inpainter.inpaint(image, translated_blocks)

        # 4. 渲染
        logger.info("步骤 4/4: 渲染译文...")
        result = self.renderer.render(inpainted, translated_blocks)

        # 5. 保存
        if output_path is None:
            suffix = self.config.output_config.get("suffix", "_translated")
            output_path = image_path.parent / f"{image_path.stem}{suffix}{image_path.suffix}"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)
        logger.info("翻译完成! 输出: %s", output_path)

        return output_path

    def translate_batch(
        self,
        image_paths: list[str | Path],
        output_dir: str | Path | None = None,
        source_lang: str | None = None,
        target_lang: str | None = None,
    ) -> list[Path]:
        """
        批量翻译多张图片。

        Args:
            image_paths: 输入图片路径列表
            output_dir: 输出目录，None 则输出到原图同目录
            source_lang: 源语言
            target_lang: 目标语言

        Returns:
            输出文件路径列表
        """
        results = []
        for i, path in enumerate(image_paths):
            path = Path(path)
            logger.info("处理 [%d/%d]: %s", i + 1, len(image_paths), path.name)

            if output_dir:
                out = Path(output_dir) / path.name
            else:
                out = None

            try:
                result = self.translate_image(path, out, source_lang, target_lang)
                results.append(result)
            except Exception as e:
                logger.error("处理 %s 失败: %s", path.name, e)

        logger.info("批量处理完成: %d/%d 成功", len(results), len(image_paths))
        return results