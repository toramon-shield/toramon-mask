#!/usr/bin/env python3
"""sumeshi 公開ベンチ（eval2/challenge2 等）を Toramon 共通 JSONL へ変換する（診断用。主たる独立テストではない）。

元形式: {"text":..., "spans":[{"start","end","label"}]}（コードポイント位置）。
全 span を action=MASK として扱う（作者の方針。Toramon 方針との差分は評価報告で注記）。

  python scripts/convert_sumeshi_bench.py third_party/privacy-filter-jp.cpp/datasets/benchmark data/public_bench
"""
import json, sys, hashlib, datetime
from pathlib import Path

if len(sys.argv) != 3 or sys.argv[1] in ("-h", "--help"):
    sys.exit(__doc__)
src_dir = Path(sys.argv[1]); out_dir = Path(sys.argv[2]); out_dir.mkdir(parents=True, exist_ok=True)
for name in ["eval2", "challenge2", "eval", "challenge", "train"]:
    p = src_dir / f"{name}.jsonl"
    if not p.exists():
        continue
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    with open(out_dir / f"sumeshi_{name}.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            text = r["text"]
            ents = []
            for j, s in enumerate(r.get("spans", [])):
                ents.append({"start": s["start"], "end": s["end"], "text": text[s["start"]:s["end"]], "label": s["label"],
                             "action": "MASK", "entity_id": f"e{j}", "source": "human_added"})
            doc = {"id": f"sumeshi-{name}-{i:03d}", "split": "dev" if name != "challenge2" else "test",
                   "scenario_family": f"sumeshi-{name}-{i:03d}", "template_family": "sumeshi-handwritten",
                   "generator_model": "human(sumeshi)", "prompt_version": "n/a", "domain": "mixed",
                   "negative_type": "none" if ents else "no_pii", "text": text, "entities": ents,
                   "review_status": "human_verified",
                   "provenance": {"created_at": "2026-07-03", "pipeline_version": "sumeshi-benchmark", "synthetic": True,
                                  "notes": "public diagnostic benchmark; author policy (all spans MASK)"}}
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(name, len(rows), "rows ->", out_dir / f"sumeshi_{name}.jsonl")
