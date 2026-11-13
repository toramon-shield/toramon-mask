#!/usr/bin/env python3
"""本番（train / dev / test）生成ジョブを作る。scenario_families.yaml の凍結 split に従い、各 split 専用の value_pool を割り当てる。

  .venv/bin/python scripts/make_jobs.py --split train --n 30000 --out data/train/jobs.jsonl --seed 1
  .venv/bin/python scripts/make_jobs.py --split dev --n 3000 --out data/dev/jobs.jsonl --seed 2
  .venv/bin/python scripts/make_jobs.py --split test --n 5000 --out <評価領域>/jobs.jsonl --seed 3   # 評価担当のみ実行

--routes の系統へほぼ均等に割り当て（既定 3 系統）（spec を 3 等分し、系統ごとに別 spec を割り当てる。パイロットと違い同一 spec を 3 系統で重複生成しない）。
"""
import argparse, json, random, yaml
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT / "scripts"))
from make_pilot_jobs import MASK_KINDS, KIND_WEIGHTS, KEEP_KINDS, LENGTH_BANDS, HARD, NEG  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True, choices=["train", "dev", "test"]); ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--neg-rate", type=float, default=0.25)
    ap.add_argument("--natural-rate", type=float, default=0.0, help="自然文直接生成トラックの比率（dev/test で 0.1 目安）")
    ap.add_argument("--routes", default="gpt,gemini,claude", help="生成担当の経路（カンマ区切り、順に均等割り当て）。gpt / gemini / claude / local")
    a = ap.parse_args()
    fam = [f for f in yaml.safe_load(open(ROOT / "configs/scenario_families.yaml", encoding="utf-8"))["families"] if f["split"] == a.split]
    rng = random.Random(a.seed)
    routes = [r.strip() for r in a.routes.split(",") if r.strip()]
    jobs = []
    for i in range(a.n):
        f = rng.choice(fam)
        r = rng.random()
        neg = "none" if r > a.neg_rate else ("no_pii" if r < a.neg_rate * 0.45 else "pii_lookalike")
        lb = rng.choices(LENGTH_BANDS, weights=[w for _, _, w in LENGTH_BANDS])[0]
        n_people = 0 if neg != "none" else rng.choice([1, 1, 1, 2, 2, 3])
        kinds = []
        if neg == "none":
            kinds = sorted(set(rng.choices(MASK_KINDS, weights=[KIND_WEIGHTS[x] for x in MASK_KINDS], k=rng.randint(2, 6))))
            if not any(x.startswith("PERSON") for x in kinds): kinds.insert(0, "PERSON")
        keep = sorted(set(rng.sample(KEEP_KINDS, rng.randint(1, 4)))) if (neg != "none" or rng.random() < 0.6) else []
        hard = rng.sample(HARD, rng.choice([1, 1, 2, 3])) if neg == "none" else []
        negax = rng.sample(NEG, rng.randint(1, 3)) if neg != "none" else []
        jobs.append({"job_id": f"{a.split}-{i:06d}", "spec_id": f"{a.split}-{i:06d}", "scenario_family": f["id"], "domain": f["domain"], "family_desc": f["description"],
                     "length_band": lb[0], "length_hint": lb[1], "negative_type": neg, "n_people": n_people, "mask_kinds": kinds, "keep_kinds": keep,
                     "difficulty_axes": hard, "negative_axes": negax, "family_split": a.split, "value_pool": f"pool_{a.split}",
                     "generator_route": routes[i % len(routes)], "split": a.split, "prompt_version": "gen_v1",
                     "track": "natural" if rng.random() < a.natural_rate else "placeholder"})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fo:
        for j in jobs: fo.write(json.dumps(j, ensure_ascii=False) + "\n")
    print(len(jobs), Counter(j["generator_route"] for j in jobs), Counter(j["negative_type"] for j in jobs), Counter(j["length_band"] for j in jobs), "families", len(fam))


if __name__ == "__main__":
    main()
