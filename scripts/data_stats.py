#!/usr/bin/env python3
"""データセットの分布統計（データカード用）: 件数、生成元、業務分野、長さ帯、負例、PII 密度、カテゴリ/action、難易度軸、レビュー状態。
  python scripts/data_stats.py data/pilot/filled.jsonl [--out runs/pilot/data_stats.json]
"""
import argparse, json, re, statistics
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--out")
a = ap.parse_args()
rows = [json.loads(l) for p in a.files for l in open(p, encoding="utf-8") if l.strip()]
C = lambda key: dict(Counter(r.get(key) for r in rows))
lens = [len(r["text"]) for r in rows]
mask_counts = [sum(1 for e in r["entities"] if e["action"] == "MASK") for r in rows]
axes = Counter()
for r in rows:
    m = re.search(r"difficulty_axes=\[(.*?)\]", r.get("provenance", {}).get("notes", ""))
    if m:
        for x in re.findall(r"'([a-z_]+)'", m.group(1)): axes[x] += 1
st = {
    "n_docs": len(rows), "generator_model": C("generator_model"), "domain": C("domain"), "length_band": C("length_band"), "negative_type": C("negative_type"),
    "difficulty": C("difficulty"), "review_status": C("review_status"), "split": C("split"),
    "chars": {"median": statistics.median(lens), "p10": sorted(lens)[len(lens)//10], "p90": sorted(lens)[int(len(lens)*0.9)], "max": max(lens), "total": sum(lens)},
    "mask_spans_per_doc": {"mean": round(statistics.fmean(mask_counts), 2), "median": statistics.median(mask_counts), "max": max(mask_counts)},
    "pii_density_per_1000chars": round(1000 * sum(mask_counts) / sum(lens), 2),
    "entities_by_label_action": dict(Counter(f"{e['label']}:{e['action']}" for r in rows for e in r["entities"])),
    "entity_subtypes": dict(Counter(e.get("subtype") for r in rows for e in r["entities"] if e.get("subtype")).most_common(30)),
    "difficulty_axes": dict(axes.most_common()),
    "scenario_families": len({r["scenario_family"] for r in rows}),
}
js = json.dumps(st, ensure_ascii=False, indent=1)
if a.out: Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(js, encoding="utf-8")
print(js)
