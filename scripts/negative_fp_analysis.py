#!/usr/bin/env python3
"""負例（正解 MASK なし文書）での誤検知の原因分析。集計のみで本文は出力しない（locked 本文を開発側が読まない約束）。
  .venv/bin/python scripts/negative_fp_analysis.py --gold eval/locked/test.jsonl --pred runs/final/T-O_rules_all.pred.jsonl --out runs/final/negative_fp_analysis.md
"""
import argparse, json, collections
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--gold", required=True); ap.add_argument("--pred", required=True); ap.add_argument("--out", required=True); ap.add_argument("--name", default="")
a = ap.parse_args()
gold = {}
for l in open(a.gold, encoding="utf-8"):
    d = json.loads(l); gold[d["id"]] = d
pred = {json.loads(l)["id"]: json.loads(l).get("spans", []) for l in open(a.pred, encoding="utf-8") if l.strip()}
neg = {k: d for k, d in gold.items() if not any(e.get("action", "MASK") == "MASK" for e in d.get("entities", []))}
by_label = collections.Counter(); by_route = collections.Counter(); by_route_n = collections.Counter(); by_type = collections.Counter(); by_type_n = collections.Counter()
by_len = collections.Counter(); keep_hit = collections.Counter(); review_hit = collections.Counter(); docs_fp = 0; n_sp = 0
for k, d in neg.items():
    sp = pred.get(k, []); by_route_n[d.get("generator_route", "?")] += 1; by_type_n[d.get("negative_type", "none")] += 1
    if sp: docs_fp += 1; by_route[d.get("generator_route", "?")] += 1; by_type[d.get("negative_type", "none")] += 1
    ents = d.get("entities", [])
    for s in sp:
        n_sp += 1; by_label[s["label"]] += 1; L = s["end"] - s["start"]; by_len["1-3" if L <= 3 else "4-8" if L <= 8 else "9-20" if L <= 20 else "21+"] += 1
        hit = [e for e in ents if e["start"] < s["end"] and s["start"] < e["end"]]
        if hit:
            if any(e.get("action") == "KEEP" for e in hit): keep_hit[s["label"]] += 1
            if any(e.get("action") == "REVIEW" for e in hit): review_hit[s["label"]] += 1
lines = [f"# 負例誤検知の分析（{a.name or Path(a.pred).name}、{a.gold}）", "", f"負例文書 {len(neg)}、誤検知のある文書 {docs_fp}（{docs_fp/len(neg)*100:.1f}%）、誤検知 span {n_sp}。本文は出力しない。", "",
         "## 誤検知 span のラベル別（うち正解の KEEP span と重なるもの / REVIEW span と重なるもの）", "", "| ラベル | span 数 | KEEP と重なる | REVIEW と重なる | 正解に span なし |", "|---|---|---|---|---|"]
for lb, n in by_label.most_common(): lines.append(f"| {lb} | {n} | {keep_hit[lb]} | {review_hit[lb]} | {n - keep_hit[lb] - review_hit[lb]} |")
lines += ["", "## 生成系統別（誤検知のある文書 / 負例文書）", "", "| 系統 | 文書 | 誤検知あり | 率 |", "|---|---|---|---|"]
for r, n in sorted(by_route_n.items()): lines.append(f"| {r} | {n} | {by_route[r]} | {by_route[r]/n*100:.1f}% |")
lines += ["", "## 負例種別", "", "| 種別 | 文書 | 誤検知あり | 率 |", "|---|---|---|---|"]
for r, n in sorted(by_type_n.items()): lines.append(f"| {r} | {n} | {by_type[r]} | {by_type[r]/n*100:.1f}% |")
lines += ["", "## 誤検知 span の長さ（文字）", "", "| 長さ | span 数 |", "|---|---|"] + [f"| {k} | {by_len[k]} |" for k in ["1-3", "4-8", "9-20", "21+"]]
Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8"); print("\n".join(lines))
