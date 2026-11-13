"""リポジトリ内外のパスを 1 か所で決める。

絶対パスをコードに書かず、次の環境変数で上書きできるようにする。

- TORAMON_MODELS_DIR: 元モデル（OpenAI privacy-filter / sumeshi GGUF など）を置いたディレクトリ。
  既定はリポジトリ直下の ``models/``。
- TORAMON_PFJP_DIR: privacy-filter-jp.cpp のチェックアウト先。既定は ``third_party/privacy-filter-jp.cpp``。
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("TORAMON_MODELS_DIR", str(REPO_ROOT / "models")))
PFJP_DIR = Path(os.environ.get("TORAMON_PFJP_DIR", str(REPO_ROOT / "third_party" / "privacy-filter-jp.cpp")))

# よく使う元モデルの既定配置（README の「モデルの入手」と同じ）
OPF_DIR = MODELS_DIR / "openai-privacy-filter"
SUMESHI_GGUF_DIR = MODELS_DIR / "sumeshi-privacy-filter-jp-GGUF"
SUMESHI_F16_GGUF = SUMESHI_GGUF_DIR / "privacy-filter-jp-f16.gguf"
SUMESHI_HF_DIR = MODELS_DIR / "sumeshi-privacy-filter-jp-hf-from-f16"
