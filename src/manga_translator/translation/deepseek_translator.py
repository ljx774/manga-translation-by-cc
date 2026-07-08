"""DeepSeek 翻译引擎"""

import json
import logging
import os

from openai import OpenAI

from manga_translator.translation.base import BaseTranslator
from manga_translator.types import TextBlock, TranslatedBlock

logger = logging.getLogger(__name__)

# 漫画翻译专用 system prompt
MANGA_TRANSLATION_PROMPT = """你是一个专业的漫画翻译助手。请将以下漫画对话从 {source_lang} 翻译成 {target_lang}。

翻译要求：
1. 这是漫画中的对话，要保持口语化、自然流畅
2. 保留原文的语气和情感（惊讶、愤怒、害羞等）
3. 对于角色名、拟声词、特殊招式名等专有名词，酌情保留或音译
4. 翻译要简洁，适合放在漫画气泡中
5. 如果原文已经是 {target_lang}，则直接返回原文

输入是一段 JSON 数组，每个元素有 id 和 text 字段。请返回相同格式的 JSON 数组，将 text 字段替换为翻译结果。

输入示例：
[{"id": 0, "text": "こんにちは"}, {"id": 1, "text": "お元気ですか？"}]

请只返回翻译后的 JSON 数组，不要包含任何其他内容。"""


class DeepSeekTranslator(BaseTranslator):
    """基于 DeepSeek API 的翻译引擎

    DeepSeek API 兼容 OpenAI SDK，只需设置 base_url 和 api_key。
    官网: https://platform.deepseek.com
    """

    BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        """
        Args:
            model: 模型名称 (deepseek-chat, deepseek-reasoner 等)
            api_key: API key，不传则从环境变量 DEEPSEEK_API_KEY 读取
            base_url: API 地址，不传则使用默认 https://api.deepseek.com
        """
        self.model = model
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or self.BASE_URL

        if not self.api_key:
            raise ValueError(
                "DeepSeek API key 未设置。请设置环境变量 DEEPSEEK_API_KEY "
                "或通过参数传入 api_key。获取地址: https://platform.deepseek.com/api_keys"
            )

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def translate(
        self, blocks: list[TextBlock], source_lang: str, target_lang: str
    ) -> list[TranslatedBlock]:
        """
        批量翻译文本块。

        Args:
            blocks: 待翻译的文本块
            source_lang: 源语言 (ja, zh, en, ko 等)
            target_lang: 目标语言

        Returns:
            翻译后的文本块列表
        """
        if not blocks:
            return []

        logger.info(
            "使用 DeepSeek (%s) 翻译 %d 个文本块: %s → %s",
            self.model,
            len(blocks),
            source_lang,
            target_lang,
        )

        # 构建输入
        input_items = [
            {"id": i, "text": block.text} for i, block in enumerate(blocks)
        ]
        input_json = json.dumps(input_items, ensure_ascii=False)

        system_prompt = MANGA_TRANSLATION_PROMPT.format(
            source_lang=source_lang, target_lang=target_lang
        )

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_json},
            ],
            temperature=0.3,
        )

        content = response.choices[0].message.content or "[]"

        # 解析返回结果
        try:
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            translated_items = json.loads(content)
        except json.JSONDecodeError:
            logger.error("无法解析翻译结果: %s", content[:200])
            raise RuntimeError(f"翻译 API 返回了无法解析的内容: {content[:200]}")

        # 构建结果
        result = []
        for item in translated_items:
            idx = item["id"]
            if idx < len(blocks):
                result.append(
                    TranslatedBlock(
                        bbox=blocks[idx].bbox,
                        original=blocks[idx].text,
                        translated=item["text"],
                    )
                )

        logger.info("翻译完成: %d 个文本块", len(result))
        return result