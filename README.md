# Manga Translator - 漫画 OCR 翻译工具

自动检测漫画/图片中的文字，翻译后嵌入回图片中。

## 功能特性

- 🔍 **OCR 文字检测**：基于 EasyOCR，支持日文/中文/英文/韩文等多种语言
- 🤖 **AI 翻译**：支持 OpenAI (GPT-4o) 和 Claude API，理解漫画语境
- 🎨 **文字擦除**：自动检测背景色并填充原文区域
- 📝 **译文嵌入**：自动计算字体大小、换行和居中，将译文渲染回图片
- 📦 **批量处理**：支持单张图片和整个目录

## 安装

```bash
# 1. 克隆项目
git clone <repo-url>
cd manga-translation

# 2. 安装依赖（推荐使用虚拟环境）
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# 3. 设置 API Key（二选一）
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

## 快速开始

```bash
# 翻译日文漫画到中文
manga-translate translate page.jpg -o page_zh.jpg -s ja -t zh

# 使用 Claude 翻译
manga-translate translate page.jpg --engine claude --model claude-sonnet-5

# 批量处理目录
manga-translate translate ./manga_pages/ -o ./output/ -s ja -t zh

# 仅运行 OCR 查看识别结果
manga-translate ocr page.jpg -s ja
```

## 配置

默认配置文件位于 `config/default.yaml`，可通过 `-c` 指定自定义配置：

```yaml
# OCR 设置
ocr:
  source_lang: ja      # 源语言
  gpu: true            # 是否使用 GPU
  min_confidence: 0.5  # 最低置信度

# 翻译设置
translation:
  engine: openai       # openai 或 claude
  model: gpt-4o        # 模型名称

# 擦除设置
inpainting:
  padding: 4           # 擦除区域扩展像素
  blur_radius: 3       # 边缘模糊半径

# 渲染设置
rendering:
  font_color: [0, 0, 0]  # 字体颜色
  min_font_size: 10
  max_font_size: 32
```

## 项目结构

```
src/manga_translator/
├── cli.py                  # CLI 入口
├── pipeline.py             # 流程编排
├── config.py               # 配置管理
├── types.py                # 共享数据类型
├── ocr/
│   ├── base.py             # OCR 抽象接口
│   └── easyocr_engine.py   # EasyOCR 实现
├── translation/
│   ├── base.py             # 翻译抽象接口
│   ├── openai_translator.py
│   └── claude_translator.py
├── inpainting/
│   ├── base.py             # 擦除抽象接口
│   └── simple_inpaint.py   # 背景色填充实现
└── rendering/
    ├── base.py             # 渲染抽象接口
    └── text_renderer.py    # PIL 文字渲染
```

## 注意事项

- 首次运行 EasyOCR 会自动下载模型文件（约 100-300MB）
- 需要有效的 OpenAI 或 Anthropic API Key
- 建议安装 `fonts-noto-cjk` 以获得最佳 CJK 字体渲染效果
- CPU 模式下 OCR 初次加载较慢，GPU 模式需要 CUDA 支持