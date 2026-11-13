#!/usr/bin/env bash
# 外部リポジトリ 2 つを、検証済みの固定コミットで third_party/ に取得し、Toramon 用パッチを当てる。
#
#   scripts/setup_third_party.sh
#
# - privacy-filter-jp.cpp（sumeshi、MIT）: GGUF 推論ランタイム。ggml サブモジュールも取得する。
#   （作者の非公開データ用サブモジュール datasets/jp-data は不要なので取得しない）
# - openai/privacy-filter（Apache-2.0）: 公式デコーダ・学習 CLI（opf）。Docker イメージに組み込む。
#
# 取得元は環境変数で差し替え可能（ミラーを使う場合など）: PFJP_URL / OPF_URL
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TP=$ROOT/third_party
mkdir -p "$TP"

PFJP_URL=${PFJP_URL:-https://github.com/sumeshi/privacy-filter-jp.cpp.git}
PFJP_COMMIT=fc5ea7dd04ecac978c377cb77fe144dcab71379a
OPF_URL=${OPF_URL:-https://github.com/openai/privacy-filter.git}
OPF_COMMIT=f7f00ca7fb869683eb732c010299d901457f19c3

fetch() {  # fetch <url> <dir> <commit>
  local url=$1 dir=$2 commit=$3
  if [ ! -d "$dir/.git" ]; then
    echo "== clone $url"
    git clone --quiet "$url" "$dir"
  fi
  git -C "$dir" fetch --quiet origin "$commit" 2>/dev/null || true
  git -C "$dir" checkout --quiet "$commit"
  echo "   $dir @ $(git -C "$dir" rev-parse --short HEAD)"
}

fetch "$PFJP_URL" "$TP/privacy-filter-jp.cpp" "$PFJP_COMMIT"
git -C "$TP/privacy-filter-jp.cpp" submodule update --init --quiet ggml
fetch "$OPF_URL" "$TP/openai-privacy-filter-repo" "$OPF_COMMIT"

echo "== apply patches"
for p in "$ROOT"/env/patches/*.patch; do
  if git -C "$TP/privacy-filter-jp.cpp" apply --check "$p" 2>/dev/null; then
    git -C "$TP/privacy-filter-jp.cpp" apply "$p"; echo "   applied $(basename "$p")"
  elif git -C "$TP/privacy-filter-jp.cpp" apply --reverse --check "$p" 2>/dev/null; then
    echo "   already applied $(basename "$p")"
  else
    echo "   パッチを当てられません: $p" >&2; exit 1
  fi
done
echo "完了。次は env/build_pfshared.sh（GPU なしの PC は PF_CUDA=OFF env/build_pfshared.sh）"
