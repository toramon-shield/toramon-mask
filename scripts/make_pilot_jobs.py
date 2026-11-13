#!/usr/bin/env python3
"""パイロット 300 件（各生成元 100 件）のジョブ仕様を作る。開発側のみで使用（split=pilot、train/dev の family と pool_train）。

各ジョブ: scenario_family, template 指示（doc_type の体裁）, 業務分野, 文書長帯, 使う PII 種類と人数, 難易度軸, 負例条件, 生成元。
3 系統に同じ仕様分布を割り当てる（仕様は同じ、文書は独立生成）。
"""
import json, random, sys, yaml, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
fam = yaml.safe_load(open(ROOT / "configs/scenario_families.yaml", encoding="utf-8"))["families"]
axes = yaml.safe_load(open(ROOT / "configs/difficulty_axes.yaml", encoding="utf-8"))
HARD = list(axes["hard_axes"]); NEG = list(axes["negative_axes"])
MASK_KINDS = ["PERSON", "PERSON_KANA", "PERSON_FAMILY", "PERSON_ROMAJI", "ADDRESS", "ADDRESS_PARTIAL", "POSTAL", "EMAIL", "PHONE",
              "PHONE_INTL", "FAX", "DATE_BIRTH", "DATE_PERSONAL", "ACCOUNT", "MYNUMBER", "LICENSE", "PASSPORT", "PENSION", "INSURANCE",
              "MEMBER_ID", "ORDER_ID", "URL_PERSONAL", "URL_TRACK", "SECRET_PASSWORD", "SECRET_APIKEY", "SECRET_TOKEN"]
# カテゴリ配分（goal/data_design §3）に沿った重み
KIND_WEIGHTS = {"PERSON": 10, "PERSON_KANA": 3, "PERSON_FAMILY": 4, "PERSON_ROMAJI": 2, "ADDRESS": 8, "ADDRESS_PARTIAL": 2, "POSTAL": 1,
                "EMAIL": 8, "PHONE": 9, "PHONE_INTL": 1, "FAX": 1, "DATE_BIRTH": 5, "DATE_PERSONAL": 3, "ACCOUNT": 4, "MYNUMBER": 2,
                "LICENSE": 1, "PASSPORT": 1, "PENSION": 1, "INSURANCE": 1, "MEMBER_ID": 4, "ORDER_ID": 3, "URL_PERSONAL": 2, "URL_TRACK": 2,
                "SECRET_PASSWORD": 2, "SECRET_APIKEY": 2, "SECRET_TOKEN": 1}
KEEP_KINDS = ["COMPANY", "COMPANY_ADDRESS", "COMPANY_PHONE", "SHARED_EMAIL", "PUBLIC_URL", "DATE_BUSINESS", "PRODUCT_CODE", "VERSION",
              "ERROR_CODE", "AMOUNT", "INVOICE_NO", "CORP_NUMBER", "PLACE", "PUBLIC_PERSON"]
LENGTH_BANDS = [("short", "150〜300文字", 0.2), ("normal", "400〜1,200文字", 0.65), ("long", "2,000〜3,500文字", 0.15)]


def main(n_per_gen: int = 100, seed: int = 1109):
    rng = random.Random(seed)
    fams = [f for f in fam if f["split"] in ("train", "dev")]  # test family は開発側に出さない
    jobs = []
    specs = []
    for i in range(n_per_gen):
        f = fams[i % len(fams)] if i < len(fams) else rng.choice(fams)
        r = rng.random()
        neg = "none" if r > 0.25 else ("no_pii" if r < 0.12 else "pii_lookalike")
        lb = rng.choices(LENGTH_BANDS, weights=[w for _, _, w in LENGTH_BANDS])[0]
        n_people = 0 if neg != "none" else rng.choice([1, 1, 1, 2, 2, 3])
        kinds = []
        if neg == "none":
            k = rng.randint(2, 6)
            kinds = sorted(set(rng.choices(MASK_KINDS, weights=[KIND_WEIGHTS[x] for x in MASK_KINDS], k=k)))
            if not any(x.startswith("PERSON") for x in kinds):
                kinds.insert(0, "PERSON")
        keep = sorted(set(rng.sample(KEEP_KINDS, rng.randint(1, 4)))) if (neg != "none" or rng.random() < 0.6) else []
        hard = rng.sample(HARD, rng.choice([1, 1, 2, 3])) if neg == "none" else []
        negax = rng.sample(NEG, rng.randint(1, 3)) if neg != "none" else []
        specs.append({"spec_id": f"pilot-{i:03d}", "scenario_family": f["id"], "domain": f["domain"], "family_desc": f["description"],
                      "length_band": lb[0], "length_hint": lb[1], "negative_type": neg, "n_people": n_people,
                      "mask_kinds": kinds, "keep_kinds": keep, "difficulty_axes": hard, "negative_axes": negax,
                      "family_split": f["split"], "value_pool": "pool_train"})
    for gen in ("gpt", "gemini", "claude"):
        for s in specs:
            jobs.append({**s, "job_id": f"{s['spec_id']}-{gen}", "generator_route": gen, "split": "pilot", "prompt_version": "gen_v1"})
    out = ROOT / "data/pilot/jobs.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    from collections import Counter
    print(len(jobs), "jobs;", Counter(j["negative_type"] for j in specs), Counter(j["length_band"] for j in specs), Counter(j["domain"] for j in specs))


if __name__ == "__main__":
    main()
