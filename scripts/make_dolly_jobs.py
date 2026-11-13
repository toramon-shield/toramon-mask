#!/usr/bin/env python3
"""dolly-15k-ja から PII 埋め込みジョブとサブエージェント用バッチを作る。
  python scripts/make_dolly_jobs.py --n 160 --seed 11 --batch-size 16 [--offset 0]
- 各レコードに Toramon ジョブ（job_id dolly-NNNNNN、scenario_family dolly.<category>、value_pool pool_train、family_split train）を割り当てる。
- 既存 jobs.jsonl があれば追記（dolly_index の重複は避ける）。バッチ番号は既存の続き。
"""
import argparse, json, random, re, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/external/dolly_ja_raw/databricks-dolly-15k-ja.jsonl"
OUT = ROOT / "data/dolly"
MASK_POOL = ["PERSON", "PERSON", "PERSON", "EMAIL", "PHONE", "ADDRESS", "MEMBER_ID", "ORDER_ID", "DATE_BIRTH", "ACCOUNT", "PERSON_KANA", "PERSON_ROMAJI", "URL_PERSONAL", "LICENSE", "INSURANCE", "SECRET_APIKEY", "PHONE_INTL", "POSTAL", "DATE_PERSONAL", "URL_TRACK"]
KEEP_POOL = ["COMPANY", "COMPANY_PHONE", "SHARED_EMAIL", "PUBLIC_URL", "DATE_BUSINESS", "AMOUNT", "PRODUCT_CODE", "PLACE", "INVOICE_NO", "CORP_NUMBER", "PUBLIC_PERSON", "ERROR_CODE", "VERSION"]
AXES = ["kyujitai", "foreign_name", "kana_name", "joint_name", "furigana", "fullwidth_mix", "phone_nosep", "phone_intl", "wareki", "address_abbrev", "same_surname", "quotation", "en_ja_mixed", "key_value", "label_word_adjacent", "honorific_variants", "zero_width", "emoji"]
NEG_AXES = ["amounts", "product_codes", "public_info", "non_personal_dates", "place_names", "company_contacts"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=160); ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--batch-size", type=int, default=16); ap.add_argument("--route", default="claude", help="generator_route（claude=コーディングエージェントのサブエージェント、local=OpenAI 互換ローカルサーバー）"); ap.add_argument("--no-batches", action="store_true"); a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "raw_subagent").mkdir(exist_ok=True); (OUT / "raw").mkdir(exist_ok=True)
    jobs_path = OUT / "jobs.jsonl"
    existing = [json.loads(l) for l in open(jobs_path, encoding="utf-8")] if jobs_path.exists() else []
    used = {j["dolly_index"] for j in existing}
    # カテゴリ層化サンプリング
    by_cat = {}
    for i, r in enumerate(rows):
        if i not in used: by_cat.setdefault(r["category"], []).append(i)
    cats = sorted(by_cat); per = max(1, a.n // len(cats)); picked = []
    for c in cats:
        idx = by_cat[c]; rng.shuffle(idx); picked += idx[:per]
    rest = [i for c in cats for i in by_cat[c][per:]]; rng.shuffle(rest); picked += rest[:max(0, a.n - len(picked))]
    picked = picked[:a.n]
    start = len(existing); jobs = []
    for k, i in enumerate(picked):
        r = rows[i]; jid = f"dolly-{start + k:06d}"
        u = rng.random(); neg = "none" if u < 0.85 else ("pii_lookalike" if u < 0.95 else "no_pii")
        n_people = rng.choice([1, 1, 1, 2, 2, 3])
        mk = sorted(set(rng.sample(MASK_POOL, rng.randint(2, 5)))) if neg == "none" else []
        kk = sorted(set(rng.sample(KEEP_POOL, rng.randint(1, 3) if neg == "none" else rng.randint(2, 4))))
        axes = rng.sample(AXES, rng.choice([0, 1, 1, 2])) if neg == "none" else []
        naxes = rng.sample(NEG_AXES, 2) if neg != "none" else []
        src_len = len(r["instruction"]) + len(r.get("context", "")) + len(r["response"])
        band = "short" if src_len < 150 else "normal"
        jobs.append({"job_id": jid, "spec_id": jid, "scenario_family": f"dolly.{r['category']}", "domain": "dolly_qa",
                     "family_desc": f"Dolly-ja {r['category']} の Q&A を業務文脈に置き、話題に関連する個人情報を埋め込む",
                     "length_band": band, "length_hint": "元の質問・回答の内容を全て含み、400〜1,500 文字", "negative_type": neg, "n_people": n_people,
                     "mask_kinds": mk, "keep_kinds": kk, "difficulty_axes": axes, "negative_axes": naxes, "family_split": "train", "value_pool": "pool_train",
                     "generator_route": a.route, "split": "dolly", "prompt_version": "dolly_pii_v1", "track": "placeholder", "dolly_index": i,
                     "dolly": {"instruction": r["instruction"], "context": r.get("context", ""), "response": r["response"], "category": r["category"]}})
    with open(jobs_path, "a", encoding="utf-8") as f:
        for j in jobs: f.write(json.dumps(j, ensure_ascii=False) + "\n")
    nmax = max([int(re.search(r"batch(\d+)", p).group(1)) for p in glob.glob(str(OUT / "subagent_gen_batch*.jsonl"))] or [0])
    made = []
    for b in (range(0, len(jobs), a.batch_size) if not a.no_batches else []):
        n = nmax + 1 + b // a.batch_size; p = OUT / f"subagent_gen_batch{n}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for j in jobs[b:b + a.batch_size]: f.write(json.dumps(j, ensure_ascii=False) + "\n")
        made.append(p.name)
    print(json.dumps({"added": len(jobs), "total_jobs": start + len(jobs), "batches": made, "neg": sum(1 for j in jobs if j["negative_type"] != "none")}, ensure_ascii=False))

if __name__ == "__main__":
    main()
