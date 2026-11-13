#!/usr/bin/env python3
"""JSONL を scenario_family 単位で 2 分割する（小規模実験用。本番 split は scenario_families.yaml の凍結割当を使う）。
  python scripts/split_by_family.py --inp data/pilot/filled.jsonl --out-a data/pilot/exp_train.jsonl --out-b data/pilot/exp_dev.jsonl --frac-b 0.25 --seed 0
"""
import argparse, json, random
from collections import defaultdict
ap = argparse.ArgumentParser(); ap.add_argument("--inp", required=True); ap.add_argument("--out-a", required=True); ap.add_argument("--out-b", required=True)
ap.add_argument("--frac-b", type=float, default=0.25); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip()]
fam = defaultdict(list)
for r in rows: fam[r["scenario_family"]].append(r)
keys = sorted(fam); random.Random(a.seed).shuffle(keys)
nb = max(1, round(len(keys) * a.frac_b)); b_keys = set(keys[:nb])
with open(a.out_a, "w", encoding="utf-8") as fa, open(a.out_b, "w", encoding="utf-8") as fb:
    for k in keys:
        for r in fam[k]:
            (fb if k in b_keys else fa).write(json.dumps(r, ensure_ascii=False) + "\n")
print({"families": len(keys), "a_docs": sum(len(fam[k]) for k in keys if k not in b_keys), "b_docs": sum(len(fam[k]) for k in b_keys)})
