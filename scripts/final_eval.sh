#!/usr/bin/env bash
# 隔離された最終評価。configs/frozen_conditions.json に記録された候補・基準モデルだけを、eval/locked/test.jsonl に対して一括評価する。
# 実行者は評価担当（開発担当とは別のジョブ）。開発側へ返すのは runs/final/report.md と集計のみ。
set -euo pipefail
cd "$(dirname "$0")/.."
FROZEN=${FROZEN:-configs/frozen_conditions.json}   # 第 2 サイクルは FROZEN=configs/frozen_conditions_cycle2.json OUT=runs/final_cycle2 で実行
OUT=${OUT:-runs/final}
GOLD=${1:-eval/locked/test.jsonl}
[ -f "$FROZEN" ] || { echo "frozen_conditions.json がありません。先に scripts/freeze_conditions.py を実行してください" >&2; exit 2; }
[ -f "$GOLD" ] || { echo "最終テスト $GOLD がありません" >&2; exit 2; }
# 凍結後にコードが変わっていないか検査
.venv/bin/python - "$FROZEN" <<'PY'
import json, sys, hashlib
from pathlib import Path
fz = json.load(open(sys.argv[1])); bad = []
def sha(p):
    h = hashlib.sha256()
    for chunk in iter(lambda: open(p, "rb").read(1 << 22), b""): h.update(chunk)
    return h.hexdigest()
import io
for rel, h in fz["artifacts"].items():
    cur = hashlib.sha256(Path(rel).read_bytes()).hexdigest()
    if cur != h: bad.append(rel)
if bad: sys.exit("凍結後に変更されたファイル: " + ", ".join(bad))
print("frozen artifacts verified:", len(fz["artifacts"]))
PY
MODELS=()
while IFS= read -r line; do MODELS+=(--model "$line"); done < <(.venv/bin/python - "$FROZEN" "$(pwd)" "${TORAMON_MODELS_DIR:-$(pwd)/models}" <<'PY'
import json, sys
fz = json.load(open(sys.argv[1]))
ROOT = sys.argv[2].rstrip("/"); MODELS = sys.argv[3].rstrip("/")
def cpath(p):  # コンテナ内パスへ（opf/hf はコンテナで実行される）
    if p.startswith(ROOT + "/"): return "/work/" + p[len(ROOT) + 1:]
    if p.startswith(MODELS + "/"): return "/models/" + p[len(MODELS) + 1:]
    return p
for name, m in fz["models"].items():
    p = m["path"]
    if p.endswith(".gguf"): print(f"{name}=gguf:{p}")
    elif name == "O": print(f"{name}=opf:{cpath(p)}")
    else:
        ls = "S" if name.startswith("T-S") else "identity"
        print(f"{name}=hf:{ls}:{cpath(p)}")
PY
)
.venv/bin/python scripts/run_comparison.py --gold "$GOLD" --out "$OUT" --title "最終評価（locked test、$(basename "$FROZEN" .json)）" "${MODELS[@]}" --configs model_only,rules_all --baseline S
echo "最終評価完了: $OUT/report.md"
