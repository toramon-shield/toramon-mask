#!/usr/bin/env bash
# 比較・学習の起点になる公開モデルを Hugging Face から取得する（検証済みリビジョンに固定）。
#
#   scripts/download_models.sh            # sumeshi GGUF のみ（約 4.5 GB。まず試すならこれだけで十分）
#   scripts/download_models.sh --all      # OpenAI privacy-filter も取得（さらに約 5.6 GB）
#
# 保存先: TORAMON_MODELS_DIR（既定はリポジトリ直下の models/）
# 必要なもの: huggingface_hub の CLI（`hf`。ホスト venv に入っている: .venv/bin/hf）
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODELS=${TORAMON_MODELS_DIR:-$ROOT/models}
HF=${HF_CLI:-$ROOT/.venv/bin/hf}
command -v "$HF" >/dev/null 2>&1 || HF=hf
mkdir -p "$MODELS"

echo "== sumeshi/privacy-filter-jp-GGUF（Apache-2.0）"
"$HF" download sumeshi/privacy-filter-jp-GGUF --revision 0c21b7fbebadb729d99893b4b58275c87f128e93 \
  --local-dir "$MODELS/sumeshi-privacy-filter-jp-GGUF"

if [ "${1:-}" = "--all" ]; then
  echo "== openai/privacy-filter（Apache-2.0）"
  # 検証に必要なファイルだけ取得する（configs/model_assets_sha256.txt と同じ一覧）
  "$HF" download openai/privacy-filter --revision 7ffa9a043d54d1be65afb281eddf0ffbe629385b \
    --local-dir "$MODELS/openai-privacy-filter" \
    README.md config.json model.safetensors tokenizer.json tokenizer_config.json viterbi_calibration.json \
    original/config.json original/dtypes.json original/model.safetensors original/viterbi_calibration.json
fi

echo "== SHA256 を検証（configs/model_assets_sha256.txt）"
cd "$MODELS"
PREFIX='sumeshi-privacy-filter-jp-GGUF/'; [ "${1:-}" = "--all" ] && PREFIX='(sumeshi-privacy-filter-jp-GGUF|openai-privacy-filter)/'
grep -E "^[0-9a-f]{64}  $PREFIX" "$ROOT/configs/model_assets_sha256.txt" | sha256sum -c -
echo "完了: $MODELS"
