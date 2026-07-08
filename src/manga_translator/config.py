"""配置管理模块"""

import os
from pathlib import Path
from typing import Any

import yaml


class Config:
    """全局配置管理器

    加载顺序：默认配置 → 用户配置文件 → 环境变量覆盖
    """

    def __init__(self, config_path: str | Path | None = None):
        self._data: dict[str, Any] = {}

        # 1. 加载默认配置
        default_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"
        if default_path.exists():
            with open(default_path, "r", encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}

        # 2. 加载用户指定配置
        if config_path:
            config_path = Path(config_path)
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    user_config = yaml.safe_load(f) or {}
                    self._deep_update(self._data, user_config)

        # 3. 环境变量覆盖
        self._apply_env_overrides()

    def _deep_update(self, base: dict, override: dict) -> None:
        """递归合并配置"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def _apply_env_overrides(self) -> None:
        """应用环境变量覆盖"""
        env_mappings = {
            "OCR_SOURCE_LANG": "ocr.source_lang",
            "TRANSLATION_ENGINE": "translation.engine",
            "TRANSLATION_SOURCE_LANG": "translation.source_lang",
            "TRANSLATION_TARGET_LANG": "translation.target_lang",
            "TRANSLATION_MODEL": "translation.model",
            "OPENAI_API_KEY": "translation.openai_api_key",
            "ANTHROPIC_API_KEY": "translation.anthropic_api_key",
            "DEEPSEEK_API_KEY": "translation.deepseek_api_key",
        }
        for env_var, config_path in env_mappings.items():
            value = os.environ.get(env_var)
            if value:
                self._set_nested(config_path, value)

    def _set_nested(self, path: str, value: str) -> None:
        """设置嵌套配置值"""
        keys = path.split(".")
        d = self._data
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = value

    def get(self, path: str, default: Any = None) -> Any:
        """获取配置值，支持点号分隔的路径"""
        keys = path.split(".")
        d = self._data
        for key in keys:
            if isinstance(d, dict) and key in d:
                d = d[key]
            else:
                return default
        return d

    @property
    def ocr_config(self) -> dict[str, Any]:
        return self._data.get("ocr", {})

    @property
    def translation_config(self) -> dict[str, Any]:
        return self._data.get("translation", {})

    @property
    def inpainting_config(self) -> dict[str, Any]:
        return self._data.get("inpainting", {})

    @property
    def rendering_config(self) -> dict[str, Any]:
        return self._data.get("rendering", {})

    @property
    def output_config(self) -> dict[str, Any]:
        return self._data.get("output", {})