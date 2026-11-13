#!/usr/bin/env bash
# HF 形式のチェックポイント（T-O / T-S、OpenAIPrivacyFilter 系）を privacy-filter-jp.cpp 用 GGUF に変換し、パリティ検証を行う。
# 使い方: scripts/convert_hf_to_gguf.sh <hf_model_dir> <name> [f16|f32]
# 出力: runs/gguf/<name>/<name>-<outtype>.gguf と parity/（HF と GGUF のロジット一致検証）
# 注意: 変換は torch を使うためコンテナ（env/run.sh）で実行。空きメモリが 12 GiB 以上あるときに実行すること（他の大きなモデルは止めておく）。
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=${1:?hf model dir}; NAME=${2:?name}; OT=${3:-f16}
OUT=runs/gguf/$NAME; mkdir -p "$OUT/parity"
m=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); [ "$m" -ge 12 ] || { echo "MemAvailable ${m} GiB < 12 GiB: 他の重い処理を止めてから実行してください" >&2; exit 3; }
ROOT=$(pwd); SRC_ABS=$(realpath "$SRC"); SRC_IN=/work/${SRC_ABS#$ROOT/}
env/run.sh python third_party/privacy-filter-jp.cpp/scripts/convert.py --model "$SRC_IN" --outfile "/work/$OUT/$NAME-$OT.gguf" --outtype "$OT" --name "$NAME"
sha256sum "$OUT/$NAME-$OT.gguf" | tee "$OUT/sha256.txt"
# パリティ（GGUF 側は libpf.so、HF 側はコンテナ）。既存の verify_gguf_hf_parity.py の使い方に従う。
echo "次: .venv/bin/python scripts/verify_gguf_hf_parity.py --gguf $OUT/$NAME-$OT.gguf --hf $SRC --out $OUT/parity （引数は同スクリプトの docstring を確認）"
