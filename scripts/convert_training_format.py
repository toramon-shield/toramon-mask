#!/usr/bin/env python3
"""Toramon JSONL → 学習用形式へ変換する。

  --to opf      : OpenAI opf train 形式 {"text", "spans": {"label: text": [[s,e],...]}, "info": {...}}（action=MASK のみ教師にする）
  --to sumeshi  : sumeshi finetune_jp.py 形式 {"text", "spans": [{"start","end","label"}]}
KEEP span は教師に含めない（= O ラベル）。REVIEW span を含む文書は除外（--keep-review で含めるが REVIEW は O 扱い）。
位置はいずれもコードポイント。
"""
import argparse, json, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--to", required=True, choices=["opf", "sumeshi"]); ap.add_argument("--keep-review", action="store_true")
    ap.add_argument("--min-review-status", default="auto_reviewed", choices=["unreviewed", "auto_reviewed", "human_verified"])
    a = ap.parse_args()
    order = {"unreviewed": 0, "auto_reviewed": 1, "auto_reviewed_disagreement": 0, "human_verified": 2, "quarantined": -1}
    n_in = n_out = n_skip_review = n_skip_status = 0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for p in a.inp:
            for line in open(p, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line); n_in += 1
                if order.get(r.get("review_status", "unreviewed"), 0) < order[a.min_review_status]:
                    n_skip_status += 1; continue
                ents = r.get("entities", [])
                if any(e["action"] == "REVIEW" for e in ents) and not a.keep_review:
                    n_skip_review += 1; continue
                mask = [e for e in ents if e["action"] == "MASK" and not e["label"].startswith("ext_")]
                if a.to == "opf":
                    spans: dict = {}
                    for e in mask:
                        spans.setdefault(f"{e['label']}: {e['text']}", []).append([e["start"], e["end"]])
                    f.write(json.dumps({"text": r["text"], "spans": spans, "info": {"id": r["id"], "source": "toramon", "split": r.get("split")}}, ensure_ascii=False) + "\n")
                else:
                    f.write(json.dumps({"text": r["text"], "spans": [{"start": e["start"], "end": e["end"], "label": e["label"]} for e in mask]}, ensure_ascii=False) + "\n")
                n_out += 1
    print(json.dumps({"in": n_in, "out": n_out, "skipped_review": n_skip_review, "skipped_status": n_skip_status}))


if __name__ == "__main__":
    main()
