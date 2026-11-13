#!/usr/bin/env python3
"""報告用のマスク前後サンプルを dev_v1（開発側が読んでよい分割）から作る。locked test の本文は使わない。
  .venv/bin/python scripts/make_samples.py --gold data/dev/dev_v1_749.jsonl --preds S=runs/dev_compare_v4/S_rules_all.pred.jsonl T-O=runs/dev_compare_v4/T-O-v6_rules_all.pred.jsonl --n 4 --out docs/samples_dev.md
"""
import argparse, json, random
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--gold", required=True); ap.add_argument("--preds", nargs="+", required=True); ap.add_argument("--n", type=int, default=4)
ap.add_argument("--seed", type=int, default=7); ap.add_argument("--maxlen", type=int, default=420); ap.add_argument("--out", required=True)
a = ap.parse_args()
gold = {json.loads(l)["id"]: json.loads(l) for l in open(a.gold, encoding="utf-8") if l.strip()}
preds = {}
for kv in a.preds:
    k, p = kv.split("=", 1); preds[k] = {json.loads(l)["id"]: json.loads(l).get("spans", []) for l in open(p, encoding="utf-8") if l.strip()}
def mask(text, spans, use_action=False):
    out, i = [], 0
    for s in sorted(spans, key=lambda s: s["start"]):
        if s["start"] < i: continue
        if use_action and s.get("action") == "KEEP": continue
        out.append(text[i:s["start"]]); out.append(f"[{s['label'].upper()}]"); i = s["end"]
    out.append(text[i:]); return "".join(out)
rng = random.Random(a.seed)
cands = [d for d in gold.values() if d.get("entities") and 200 <= len(d["text"]) <= a.maxlen and d.get("negative_type") in (None, "none")]
pick = rng.sample(cands, min(a.n, len(cands)))
lines = [f"# マスク前後サンプル（dev_v1、{len(pick)} 文書、合成データ）", "", "正解は Toramon 保護方針（KEEP は残す）。モデル欄は共通ルール R v0.4 併用の出力。`[LABEL]` は伏せ字。", ""]
for d in pick:
    lines += [f"## {d['id']}（{d.get('domain','?')} / {d.get('scenario_family','?')}）", "", "### 原文", "```", d["text"], "```", "", "### 正解（方針どおりの伏せ字）", "```", mask(d["text"], d["entities"], use_action=True), "```", ""]
    for k, pr in preds.items():
        lines += [f"### {k} の出力", "```", mask(d["text"], pr.get(d["id"], [])), "```", ""]
Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8"); print("wrote", a.out, len(pick))
