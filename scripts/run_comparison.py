#!/usr/bin/env python3
"""3 層比較の一括実行: 指定 gold に対して各モデル実体 × 構成（model_only / rules_all / 除去比較）を実行し、評価・対応付き比較・報告を生成する。

  .venv/bin/python scripts/run_comparison.py --gold data/dev/dev.jsonl --out runs/dev_compare --title "dev 比較" \
      --model S=gguf:<f16.gguf> --model O=opf:/models/openai-privacy-filter/original --model T-S=hf:S:<dir> --model T-O=hf:identity:<dir> \
      --configs model_only,rules_all,boundary_only,detect_only --baseline S
モデル指定: name=kind:path（kind: gguf | opf | hf:<label_space>）。opf/hf はコンテナ内で実行する（env/run.sh 経由で自動呼び出し）。
"""
from __future__ import annotations
import argparse, os, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def sh(cmd: list[str]) -> None:
    print("+", " ".join(cmd), file=sys.stderr, flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gold", required=True); ap.add_argument("--out", required=True); ap.add_argument("--title", required=True)
    ap.add_argument("--model", action="append", required=True); ap.add_argument("--configs", default="model_only,rules_all"); ap.add_argument("--baseline", default="S")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--device", default=os.environ.get("TORAMON_DEVICE", "cuda"), help="GGUF モデルを動かすデバイス（cpu / cuda）。opf/hf はコンテナ内の GPU で動く")
    a = ap.parse_args()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    gold_in_container = "/work/" + str(Path(a.gold).resolve().relative_to(ROOT)) if Path(a.gold).resolve().is_relative_to(ROOT) else a.gold
    a.gold = str(Path(a.gold).resolve())
    evals = []; preds: dict[tuple[str, str], str] = {}
    for spec in a.model:
        name, rest = spec.split("=", 1); kind, path = rest.split(":", 1)
        for cfg in a.configs.split(","):
            pred = out / f"{name}_{cfg}.pred.jsonl"
            if not pred.exists():
                if kind == "gguf":
                    sh([str(ROOT / ".venv/bin/python"), "scripts/run_filter.py", "--backend", "S", "--gguf", path, "--device", a.device, "--input", a.gold, "--out", str(pred), "--config", cfg])
                elif kind == "opf":
                    sh([str(ROOT / "env/run.sh"), "python", "scripts/run_filter.py", "--backend", "O", "--checkpoint", path, "--input", gold_in_container, "--out", "/work/" + str(pred.relative_to(ROOT)), "--config", cfg])
                elif kind.startswith("hf"):
                    ls, hpath = path.split(":", 1)
                    sh([str(ROOT / "env/run.sh"), "python", "scripts/run_filter.py", "--backend", "HF", "--checkpoint", hpath, "--label-space", ls, "--input", gold_in_container, "--out", "/work/" + str(pred.relative_to(ROOT)), "--config", cfg])
                else:
                    raise SystemExit(f"unknown kind {kind}")
            ev = out / f"{name}_{cfg}.eval.json"
            sh([str(ROOT / ".venv/bin/python"), "scripts/evaluate.py", "--gold", a.gold, "--pred", str(pred), "--out", str(ev)])
            sh([str(ROOT / ".venv/bin/python"), "scripts/error_analysis.py", "--gold", a.gold, "--pred", str(pred), "--out", str(out / f"{name}_{cfg}.errors.json")])
            evals.append(str(ev)); preds[(name, cfg)] = str(pred)
    compares = []
    for (name, cfg), p in preds.items():
        base = preds.get((a.baseline, cfg))
        if base and name != a.baseline:
            cmp_out = out / f"compare_{name}_vs_{a.baseline}_{cfg}.json"
            sh([str(ROOT / ".venv/bin/python"), "scripts/evaluate.py", "--gold", a.gold, "--pred", p, "--compare", base, "--out", str(cmp_out), "--n-boot", str(a.n_boot)])
            compares.append(str(cmp_out))
    sh([str(ROOT / ".venv/bin/python"), "scripts/make_report.py", "--title", a.title, "--evals", *evals, "--compare", *compares, "--out", str(out / "report.md")])
    print("report:", out / "report.md")


if __name__ == "__main__":
    main()
