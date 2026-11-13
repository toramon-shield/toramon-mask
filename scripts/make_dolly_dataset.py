#!/usr/bin/env python3
"""dolly split の filled.jsonl から、マスク済み本文と復号表（mask_map）付きのデータセットを作る。
  python scripts/make_dolly_dataset.py --filled data/dolly/filled.jsonl --jobs data/dolly/jobs.jsonl --out data/dolly/dataset.jsonl
各行: id, source(dolly_index/category/instruction/context/response), text(値入り原文), entities, masked_text, mask_map(トークン→元の値; 復号用), negative_type, license。
検証: mask_map で masked_text を復号すると text に一致すること（全件）。
"""
import argparse, json
from pathlib import Path
LABEL_TOKEN = {"private_person": "PERSON", "private_address": "ADDRESS", "private_email": "EMAIL", "private_phone": "PHONE", "private_date": "DATE", "account_number": "ID", "private_url": "URL", "secret": "SECRET"}

def mask(text, ents):
    """MASK span を [TOKEN_n] に置換。同じ値（同一 entity_id）は同じトークン。復号表を返す。"""
    ms = sorted([e for e in ents if e["action"] == "MASK"], key=lambda e: e["start"])
    out = []; pos = 0; token_of = {}; counters = {}; mmap = {}
    for e in ms:
        if e["start"] < pos: continue  # 重なりは先勝ち
        key = (e.get("entity_id") or f"{e['start']}", e["label"], text[e["start"]:e["end"]])
        if key not in token_of:
            base = LABEL_TOKEN.get(e["label"], "PII"); counters[base] = counters.get(base, 0) + 1
            token_of[key] = f"[{base}_{counters[base]}]"; mmap[token_of[key]] = text[e["start"]:e["end"]]
        out.append(text[pos:e["start"]]); out.append(token_of[key]); pos = e["end"]
    out.append(text[pos:])
    return "".join(out), mmap

def unmask(masked, mmap):
    # 長いトークン名から順に置換（[ID_1] と [ID_10] の衝突回避）
    for t in sorted(mmap, key=len, reverse=True): masked = masked.replace(t, mmap[t])
    return masked

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--filled", required=True); ap.add_argument("--jobs", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = {json.loads(l)["job_id"]: json.loads(l) for l in open(a.jobs, encoding="utf-8")}
    n = bad = 0; n_mask = 0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for l in open(a.filled, encoding="utf-8"):
            d = json.loads(l); j = jobs[d["id"]]
            masked, mmap = mask(d["text"], d["entities"])
            ok = unmask(masked, mmap) == d["text"]
            if not ok: bad += 1
            n += 1; n_mask += len(mmap)
            f.write(json.dumps({"id": d["id"], "source": {"dataset": "llm-jp/databricks-dolly-15k-ja", "license": "CC BY-SA 3.0", "dolly_index": j["dolly_index"], **j["dolly"]},
                                "doc_type": d.get("doc_type"), "negative_type": d["negative_type"], "text": d["text"], "entities": d["entities"],
                                "masked_text": masked, "mask_map": mmap, "roundtrip_ok": ok, "generator_model": d["generator_model"]}, ensure_ascii=False) + "\n")
    print(json.dumps({"docs": n, "roundtrip_failures": bad, "mask_tokens": n_mask, "out": a.out}, ensure_ascii=False))

if __name__ == "__main__":
    main()
