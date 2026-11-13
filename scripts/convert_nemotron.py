#!/usr/bin/env python3
"""nvidia/Nemotron-PII-Japan（CC BY 4.0）の注釈を Toramon の 8 分類 + action へ写像する。
方針（docs/annotation_policy.md）に合わせ、組織名・属性（年齢・性別・職業など）・業務日付・汎用 URL は教師にしない（KEEP/REVIEW）。
姓+名が隣接する場合は 1 span に連結。郵便番号は住所に隣接する場合のみ連結して MASK。
  .venv/bin/python scripts/convert_nemotron.py --in data/ext/nemotron-pii-japan/sample3000.jsonl --train-out ... --eval-out ... --n-train 500 --n-eval 300
"""
import argparse, json, random, collections, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toramon.spans import Span, validate_spans
MASK = {
    "last_name": "private_person", "first_name": "private_person", "user_name": "private_person",
    "street_address": "private_address",
    "email": "private_email",
    "phone_number": "private_phone", "previous_phone_number": "private_phone",
    "date_of_birth": "private_date",
    "account_number": "account_number", "credit_debit_card": "account_number", "my_number": "account_number", "medical_record_number": "account_number",
    "employee_id": "account_number", "customer_id": "account_number", "unique_id": "account_number", "health_insurer_number": "account_number",
    "health_insurance_card_symbol_number": "account_number", "public_expense_payer_number": "account_number", "vehicle_identifier": "account_number",
    "license_plate": "account_number", "device_identifier": "account_number", "mac_address": "account_number", "ipv4": "account_number", "ipv6": "account_number",
    "password": "secret", "api_key": "secret", "pin": "secret", "cvv": "secret", "http_cookie": "secret",
}
REVIEW = {"prefecture": "private_address", "city": "private_address", "postcode": "private_address", "url": "private_url", "fax_number": "private_phone", "coordinate": "private_address"}
# 教師にしない（方針上 KEEP）: date, date_time, time, company_name, domestic_company_name, bank_name, branch_name, bank_code, branch_code, swift_bic,
# medical_institution_name, educational_institution_name, public_body_name, age, gender, occupation, employment_status, education_level, language, blood_type, country
def convert(r):
    text = r["text"]; sp = sorted(r["spans"], key=lambda s: (s["start"], s["end"]))
    ents = []
    for s in sp:
        if s["label"] in MASK: ents.append({"start": s["start"], "end": s["end"], "label": MASK[s["label"]], "action": "MASK", "src": s["label"]})
        elif s["label"] in REVIEW: ents.append({"start": s["start"], "end": s["end"], "label": REVIEW[s["label"]], "action": "REVIEW", "src": s["label"]})
    # 連結: 姓+名（空白 ≤1）、郵便番号/都道府県/市区町村 + 住所（空白 ≤1）
    merged = []
    for e in ents:
        if merged:
            p = merged[-1]; gap = text[p["end"]:e["start"]]
            if len(gap) <= 1 and gap.strip() == "":
                if p["src"] == "last_name" and e["src"] == "first_name":
                    p["end"] = e["end"]; p["src"] = "full_name"; continue
                if p["label"] == "private_address" and e["label"] == "private_address" and (e["action"] == "MASK" or p["action"] == "MASK"):
                    p["end"] = e["end"]; p["action"] = "MASK"; p["src"] = "address_merged"; continue
        merged.append(dict(e))
    out = []
    for i, e in enumerate(merged):
        if e["start"] >= e["end"]: continue
        if out and e["start"] < out[-1]["end"]: continue  # 重なりは先勝ち
        out.append({"start": e["start"], "end": e["end"], "text": text[e["start"]:e["end"]], "label": e["label"], "action": e["action"], "entity_id": f"nem-{i}", "source": "nemotron:" + e["src"]})
    probs = validate_spans(text, [Span(e["start"], e["end"], e["label"], e["text"], e["action"]) for e in out])
    return {"id": "nem-" + r["uid"][:12], "text": text, "entities": out, "generator_route": "nemotron", "generator_model": "nvidia/Nemotron-PII-Japan", "domain": r["domain"],
            "document_type": r["document_type"], "negative_type": "none" if any(e["action"] == "MASK" for e in out) else "no_pii", "scenario_family": "nemotron:" + r["document_type"],
            "license": "CC BY 4.0", "review_status": "external_annotation"}, probs
ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--train-out", required=True); ap.add_argument("--eval-out", required=True)
ap.add_argument("--n-train", type=int, default=500); ap.add_argument("--n-eval", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--exclude", nargs="*", default=[], help="この jsonl に含まれる id を除外する（既存の学習/評価セットとの重複回避）")
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]; random.Random(a.seed).shuffle(rows)
excl = {json.loads(l)["id"] for p in a.exclude for l in open(p, encoding="utf-8") if l.strip()}
rows = [r for r in rows if "nem-" + r["uid"][:12] not in excl]
docs = []; bad = 0; stat = collections.Counter(); srcstat = collections.Counter()
for r in rows:
    d, probs = convert(r)
    if probs: bad += 1; continue
    for e in d["entities"]: stat[(e["label"], e["action"])] += 1; srcstat[e["source"]] += 1
    docs.append(d)
ev, tr = docs[:a.n_eval], docs[a.n_eval:a.n_eval + a.n_train]
for p, rs in ((a.train_out, tr), (a.eval_out, ev)):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for d in rs: f.write(json.dumps(d, ensure_ascii=False) + "\n")
print(json.dumps({"converted": len(docs), "span_problems_dropped": bad, "train": len(tr), "eval": len(ev), "labels": {f"{k[0]}:{k[1]}": v for k, v in sorted(stat.items())},
                  "mask_spans_per_doc": round(sum(1 for d in docs for e in d["entities"] if e["action"] == "MASK") / len(docs), 2)}, ensure_ascii=False, indent=1))
