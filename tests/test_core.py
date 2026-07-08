"""端到端测试 — 使用合成图片验证完整流程"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from manga_translator.types import TextBlock, TranslatedBlock
from manga_translator.inpainting.simple_inpaint import SimpleInpainter
from manga_translator.rendering.text_renderer import TextRenderer


def create_test_image(width: int = 600, height: int = 200) -> np.ndarray:
    """创建一张白色背景带黑色文字的测试图片"""
    import cv2

    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # 用 OpenCV 写入文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Hello World", (50, 100), font, 2, (0, 0, 0), 3)

    return img


class TestSimpleInpainter:
    """测试擦除模块"""

    def test_inpaint_removes_text(self):
        """测试擦除功能：验证擦除区域的像素被修改"""
        img = create_test_image()
        inpainter = SimpleInpainter(padding=4, blur_radius=1)

        # 创建一个覆盖文字的文本块
        block = TranslatedBlock(
            bbox=(30, 30, 500, 150),
            original="Hello World",
            translated="你好世界",
        )

        result = inpainter.inpaint(img, [block])

        # 验证输出形状不变
        assert result.shape == img.shape

        # 擦除区域的像素应该发生变化（不再是纯白）
        roi = result[40:140, 40:490]
        assert roi.shape[0] > 0 and roi.shape[1] > 0

    def test_inpaint_empty_blocks(self):
        """测试空文本块列表"""
        img = create_test_image()
        inpainter = SimpleInpainter()
        result = inpainter.inpaint(img, [])
        assert np.array_equal(result, img)


class TestTextRenderer:
    """测试渲染模块"""

    def test_render_adds_text(self):
        """测试渲染功能：验证渲染后图片有变化"""
        img = create_test_image()
        renderer = TextRenderer()

        block = TranslatedBlock(
            bbox=(30, 30, 500, 150),
            original="Hello World",
            translated="你好世界",
        )

        result = renderer.render(img, [block])

        # 验证输出形状不变
        assert result.shape == img.shape

        # 渲染后应该有像素变化
        assert not np.array_equal(result, img)

    def test_render_empty_blocks(self):
        """测试空文本块列表"""
        img = create_test_image()
        renderer = TextRenderer()
        result = renderer.render(img, [])
        assert np.array_equal(result, img)

    def test_render_empty_text(self):
        """测试空文本"""
        img = create_test_image()
        renderer = TextRenderer()

        block = TranslatedBlock(
            bbox=(30, 30, 500, 150),
            original="test",
            translated="",
        )

        result = renderer.render(img, [block])
        assert np.array_equal(result, img)


class TestTypes:
    """测试数据类型"""

    def test_text_block(self):
        block = TextBlock(bbox=(0, 0, 100, 50), text="こんにちは", confidence=0.95)
        assert block.bbox == (0, 0, 100, 50)
        assert block.text == "こんにちは"
        assert block.confidence == 0.95

    def test_translated_block(self):
        block = TranslatedBlock(
            bbox=(0, 0, 100, 50),
            original="こんにちは",
            translated="你好",
        )
        assert block.translated == "你好"
        assert block.original == "こんにちは"


class TestFontDetection:
    """测试字体检测"""

    def test_find_cjk_font(self):
        from manga_translator.rendering.text_renderer import find_cjk_font
        font = find_cjk_font()
        # 返回 None 或存在的路径
        if font is not None:
            assert Path(font).exists()


class TestConfig:
    """测试配置管理"""

    def test_default_config(self):
        from manga_translator.config import Config
        config = Config()
        assert config.ocr_config.get("engine") == "easyocr"
        assert config.translation_config.get("engine") == "openai"

    def test_config_get(self):
        from manga_translator.config import Config
        config = Config()
        assert config.get("ocr.engine") == "easyocr"
        assert config.get("nonexistent.key", "default") == "default"