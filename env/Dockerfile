# Toramon Mask 学習・評価用コンテナ（DGX Spark / aarch64 / Blackwell GB10 向け）
# ベース: NVIDIA PyTorch 26.07 (torch 2.13.0a0+nv26.7, CUDA 13.3)。digest を固定する。
# build:  docker build -f env/Dockerfile -t toramon-mask:0.1 .   （context = リポジトリ直下。先に scripts/setup_third_party.sh を実行）
FROM nvcr.io/nvidia/pytorch@sha256:3b0a5c3aa754aff7c86bc0ced07cb064ebce1048f6c12d717e2fa55766276521

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
# 依存を固定（2026-11-09 時点で pip index から確認したバージョン）。torch/numpy/safetensors/PyYAML(6.0.1, debian) 等はベースイメージ同梱のものを使う。
RUN pip install \
      transformers==5.22.2 \
      peft==0.21.2 \
      gguf==0.19.0 \
      accelerate==1.15.0 \
      tiktoken==0.14.0 \
      huggingface_hub==2.4.0 \
      jsonschema==4.26.0 \
      fastapi==0.162.2 \
      uvicorn==0.54.0 \
      pytest==9.2.0 \
  && pip list --format=freeze > /opt/pip-freeze.txt
# OpenAI 公式 opf（公式デコーダ・学習 CLI）。固定コミット f7f00ca7fb869683eb732c010299d901457f19c3
COPY third_party/openai-privacy-filter-repo /opt/opf
RUN pip install --no-deps -e /opt/opf
WORKDIR /work
