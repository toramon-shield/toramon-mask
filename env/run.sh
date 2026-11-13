#!/usr/bin/env bash
# Toramon コンテナでコマンドを実行する薄いラッパー。
#   env/run.sh python -c "import torch; print(torch.cuda.is_available())"
# マウント: /work = このリポジトリ, /models = 元モデル置き場（TORAMON_MODELS_DIR、既定は ./models）。外部通信は本番推論では不要（HF_HUB_OFFLINE=1）。
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODELS=${TORAMON_MODELS_DIR:-$ROOT/models}
IMG=${TORAMON_IMAGE:-toramon-mask:0.1}
exec docker run --rm -i --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_HUB_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1 -e OPF_CHECKPOINT=/models/openai-privacy-filter/original \
  -e PYTHONPATH=/work/src \
  -v "$ROOT":/work -v "$MODELS":/models:ro \
  -w /work "$IMG" "$@"
