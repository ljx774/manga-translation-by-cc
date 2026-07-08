"""CLI 入口"""

import logging
import sys
from pathlib import Path

import click

from manga_translator.config import Config
from manga_translator.pipeline import MangaTranslator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("manga_translator")


@click.group()
@click.version_option(version="0.1.0", prog_name="manga-translate")
def main():
    """漫画 OCR 翻译工具 — 自动检测、翻译并替换漫画中的文字"""
    pass


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "-o", "--output",
    type=click.Path(),
    default=None,
    help="输出文件路径（批量模式则为输出目录）",
)
@click.option(
    "-s", "--source",
    "source_lang",
    default=None,
    help="源语言代码 (ja, zh, en, ko 等)",
)
@click.option(
    "-t", "--target",
    "target_lang",
    default=None,
    help="目标语言代码 (zh, ja, en, ko 等)",
)
@click.option(
    "-c", "--config",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="配置文件路径",
)
@click.option(
    "--engine",
    default=None,
    help="翻译引擎 (deepseek, openai 或 claude)",
)
@click.option(
    "--model",
    default=None,
    help="翻译模型名称",
)
@click.option(
    "--gpu/--no-gpu",
    default=None,
    help="是否使用 GPU 进行 OCR",
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    default=False,
    help="显示详细日志",
)
def translate(
    input_path: str,
    output: str | None,
    source_lang: str | None,
    target_lang: str | None,
    config_path: str | None,
    engine: str | None,
    model: str | None,
    gpu: bool | None,
    verbose: bool,
):
    """翻译单张图片或整个目录

    \b
    示例:
      manga-translate translate page01.jpg -o page01_zh.jpg
      manga-translate translate page01.jpg -s ja -t zh --engine claude
      manga-translate translate ./manga_pages/ -o ./output/ -s ja -t zh
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 加载配置
    config = Config(config_path)

    # CLI 参数覆盖配置
    if engine:
        config._data.setdefault("translation", {})["engine"] = engine
    if model:
        config._data.setdefault("translation", {})["model"] = model
    if gpu is not None:
        config._data.setdefault("ocr", {})["gpu"] = gpu

    # 创建翻译器
    translator = MangaTranslator(config=config)

    input_path = Path(input_path)

    if input_path.is_dir():
        # 批量处理目录
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        image_files = sorted([
            f for f in input_path.iterdir()
            if f.suffix.lower() in image_extensions
        ])

        if not image_files:
            click.echo(f"目录 {input_path} 中没有找到图片文件", err=True)
            sys.exit(1)

        click.echo(f"找到 {len(image_files)} 张图片，开始批量翻译...")

        output_dir = Path(output) if output else None
        results = translator.translate_batch(
            image_files, output_dir, source_lang, target_lang
        )

        click.echo(f"\n批量翻译完成: {len(results)}/{len(image_files)} 成功")
        if len(results) < len(image_files):
            sys.exit(1)
    else:
        # 单张图片
        result = translator.translate_image(
            input_path, output, source_lang, target_lang
        )
        click.echo(f"翻译完成: {result}")


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "-s", "--source",
    "source_lang",
    default=None,
    help="源语言代码",
)
@click.option(
    "--gpu/--no-gpu",
    default=True,
    help="是否使用 GPU",
)
def ocr(input_path: str, source_lang: str | None, gpu: bool):
    """仅运行 OCR，输出检测到的文字"""
    config = Config()
    if gpu is not None:
        config._data.setdefault("ocr", {})["gpu"] = gpu

    from manga_translator.ocr.easyocr_engine import EasyOCREngine

    engine = EasyOCREngine(
        source_lang=source_lang or config.ocr_config.get("source_lang", "ja"),
        gpu=gpu,
    )

    input_path = Path(input_path)
    if input_path.is_dir():
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        for f in sorted(input_path.iterdir()):
            if f.suffix.lower() in image_extensions:
                blocks = engine.detect_and_recognize(str(f))
                click.echo(f"\n--- {f.name} ---")
                for i, block in enumerate(blocks):
                    click.echo(f"  [{i}] {block.text} (置信度: {block.confidence:.2f})")
    else:
        blocks = engine.detect_and_recognize(str(input_path))
        for i, block in enumerate(blocks):
            click.echo(f"[{i}] {block.text} (置信度: {block.confidence:.2f})")


if __name__ == "__main__":
    main()