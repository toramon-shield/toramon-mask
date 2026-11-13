#!/usr/bin/env python3
"""locked test の凍結: eval/locked/filled.jsonl に人手確認の補正を適用し、eval/locked/test.jsonl と test_manifest.json を書く。
補正ファイル（任意、人手確認担当が作る）: runs/locked/human_decisions.jsonl、1 行 1 判断:
  {"id": "test-000123", "op": "drop_entity",  "start": 10, "end": 14}                       # 正解から span を除く（誤ラベル）
  {"id": "...",        "op": "add_entity",   "start": 10, "end": 14, "label": "private_person", "action": "MASK"}  # 見落とし PII を追加
  {"id": "...",        "op": "set_action",   "start": 10, "end": 14, "action": "KEEP|MASK|REVIEW"}
  {"id": "...",        "op": "set_label",    "start": 10, "end": 14, "label": "private_phone"}
  {"id": "...",        "op": "set_span",     "start": 10, "end": 14, "new_start": 9, "new_end": 14}            # 境界修正
  {"id": "...",        "op": "drop_doc",     "reason": "..."}                                                   # 文書ごと除外
  {"id": "...",        "op": "verified"}                                                                        # 人手確認済み（変更なし）
text は変更しない。補正後に validate_spans で重なり・範囲外を検査し、問題があれば中止する。
使い方: .venv/bin/python scripts/freeze_locked_test.py [--decisions runs/locked/human_decisions.jsonl] [--allow-unverified]
"""
import argparse, json, hashlib, datetime, sys
from pathlib import Path
from collections import Counter
sys.path.insert(0, "src")
from toramon.spans import Span, validate_spans

ap = argparse.ArgumentParser()
ap.add_argument("--filled", default="eval/locked/filled.jsonl"); ap.add_argument("--decisions", default="runs/locked/human_decisions.jsonl")
ap.add_argument("--out", default="eval/locked/test.jsonl"); ap.add_argument("--manifest", default="eval/locked/test_manifest.json")
ap.add_argument("--allow-unverified", action="store_true", help="人手確認なしでも凍結する（報告は『合成データ上の評価』に限定）")
a = ap.parse_args()
docs = {json.loads(l)["id"]: json.loads(l) for l in open(a.filled, encoding="utf-8") if l.strip()}
dec = [json.loads(l) for l in open(a.decisions, encoding="utf-8")] if Path(a.decisions).exists() else []
if not dec and not a.allow_unverified:
    sys.exit("人手確認の判断ファイルがありません。--allow-unverified を付けると人手確認なしで凍結します（最終報告でその旨を明記）")
verified, dropped, ops = set(), set(), Counter()
human_docs = set()   # human=True の判断がある文書（人手確認済み）。それ以外の判断済み文書は llm_adjudicated
def find(ents, s, e):
    for x in ents:
        if x["start"] == s and x["end"] == e: return x
    raise SystemExit(f"span not found: {s}-{e}")
for d in dec:
    doc = docs.get(d["id"])
    if not doc: sys.exit(f"unknown id {d['id']}")
    op = d["op"]; ops[op] += 1; ents = doc["entities"]
    if d.get("human") is True or (d.get("human") is None and not d.get("adjudicated")): human_docs.add(d["id"])
    if op == "drop_doc": dropped.add(d["id"]); continue
    if op == "verified": verified.add(d["id"]); continue
    if op == "drop_entity": ents.remove(find(ents, d["start"], d["end"]))
    elif op == "add_entity": ents.append({"start": d["start"], "end": d["end"], "text": doc["text"][d["start"]:d["end"]], "label": d["label"], "action": d.get("action", "MASK"), "entity_id": f"human-{d['start']}", "source": "human"})
    elif op == "set_action": find(ents, d["start"], d["end"])["action"] = d["action"]
    elif op == "set_label": find(ents, d["start"], d["end"])["label"] = d["label"]
    elif op == "set_span":
        x = find(ents, d["start"], d["end"]); x["start"], x["end"] = d["new_start"], d["new_end"]; x["text"] = doc["text"][x["start"]:x["end"]]; x["source"] = "human_boundary"
        # 新しい範囲に完全に含まれる他の正解 span（例: 郵便番号だけの span を住所全体へ広げた場合の残り）は統合して除く
        for y in [y for y in ents if y is not x and y["start"] >= x["start"] and y["end"] <= x["end"]]:
            ents.remove(y); ops["merged_into_span"] += 1
    else: sys.exit(f"unknown op {op}")
    ents.sort(key=lambda x: x["start"]); verified.add(d["id"])
out = []
for did, doc in docs.items():
    if did in dropped: continue
    probs = validate_spans(doc["text"], [Span(e["start"], e["end"], e["label"], e["text"], e["action"]) for e in doc["entities"]])
    if probs: sys.exit(f"{did}: span problems after corrections: {probs[:3]}")
    doc["review_status"] = "human_verified" if did in human_docs else ("llm_adjudicated" if did in verified else doc.get("review_status", "unreviewed"))
    out.append(doc)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w", encoding="utf-8") as f:
    for d in out: f.write(json.dumps(d, ensure_ascii=False) + "\n")
sha = hashlib.sha256(Path(a.out).read_bytes()).hexdigest()
man = {"frozen_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "source": a.filled, "decisions": a.decisions if dec else None,
       "n_docs": len(out), "n_dropped": len(dropped), "n_human_verified": len(human_docs - dropped), "n_llm_adjudicated": len(verified - human_docs - dropped), "ops": dict(ops), "sha256": sha,
       "by_route": dict(Counter(d.get("generator_route", "unknown") for d in out)), "by_negative_type": dict(Counter(d.get("negative_type", "none") for d in out)),
       "note": "合成データ。人手確認は n_human_verified 件、LLM 裁定（Opus、方針準拠）は n_llm_adjudicated 件。凍結後は test.jsonl を変更しない"}
Path(a.manifest).write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(man, ensure_ascii=False, indent=1))
