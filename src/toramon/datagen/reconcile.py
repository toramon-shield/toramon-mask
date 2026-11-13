"""独立レビュー結果の照合。

- レビューの引用（quote + before/after）を原文に一意に同定し、span 候補にする。
- 既存 entities（プレースホルダ由来）と突き合わせ: 一致 / 境界不一致 / ラベル不一致 / レビューのみ（見落とし候補）/ entities のみ（レビュー見落とし）。
- 2 系統のレビューがともに一致 → review_status=auto_reviewed。いずれかが不一致 → auto_reviewed_disagreement（人手キュー）。
- レビューのみの MASK 指摘（プレースホルダ以外に保護対象が漏れていないか）は `reviewer_added` 候補として記録し、両レビュー一致なら REVIEW として追加（人手確認前に MASK 正解へ昇格させない）。
- 多数決だけで正解を確定しない。
"""
from __future__ import annotations

import json
from pathlib import Path

from ..spans import Span, validate_spans

COMMON = {"private_person", "private_address", "private_email", "private_phone", "private_date", "account_number", "private_url", "secret"}


def locate(text: str, quote: str, before: str = "", after: str = "") -> list[tuple[int, int]]:
    """quote の出現位置を列挙し、before/after が合う候補を優先して返す。"""
    if not quote:
        return []
    hits = []
    i = text.find(quote)
    while i != -1:
        hits.append((i, i + len(quote)))
        i = text.find(quote, i + 1)
    if len(hits) <= 1 or not (before or after):
        return hits
    scored = []
    for s, e in hits:
        sc = 0
        if before and text[max(0, s - len(before)):s].endswith(before[-min(len(before), 6):]): sc += 1
        if after and text[e:e + len(after)].startswith(after[:min(len(after), 6)]): sc += 1
        scored.append((sc, (s, e)))
    best = max(sc for sc, _ in scored)
    return [h for sc, h in scored if sc == best]


def reconcile_doc(doc: dict, reviews: list[dict]) -> dict:
    text = doc["text"]
    ents = doc.get("entities", [])
    gold = {(e["start"], e["end"]): e for e in ents}
    result = {"id": doc["id"], "n_reviews": len(reviews), "per_review": [], "agree_all": True, "reviewer_added": [], "missed_by_reviewers": []}
    found_by = {k: 0 for k in gold}
    mask_gold = {k for k, e in gold.items() if e["action"] == "MASK"}
    added_candidates: dict[tuple[int, int, str], int] = {}
    for rv in reviews:
        out = rv.get("output") or {}
        rr = {"reviewer": rv.get("reviewer_route"), "json_ok": bool(out), "unreviewable": bool(out.get("unreviewable")) if out else True,
              "matched": 0, "boundary_mismatch": 0, "label_mismatch": 0, "action_mismatch": 0, "review_only": 0, "keep_only": 0, "ambiguous_quote": 0}
        if not out or out.get("unreviewable"):
            result["agree_all"] = False; result["per_review"].append(rr); continue
        seen = set()
        for f in out.get("findings", []):
            locs = locate(text, f.get("quote", ""), f.get("before", ""), f.get("after", ""))
            if len(locs) != 1:
                rr["ambiguous_quote"] += 1
                if not locs: continue
            s, e = locs[0]
            label = f.get("label"); action = f.get("action", "MASK")
            if (s, e) in gold:
                g = gold[(s, e)]; seen.add((s, e))
                if g["label"] != label and label in COMMON: rr["label_mismatch"] += 1
                elif g["action"] != action: rr["action_mismatch"] += 1
                else: rr["matched"] += 1
                found_by[(s, e)] += 1
            else:
                ov = [k for k in gold if k[0] < e and s < k[1]]
                if ov:
                    rr["boundary_mismatch"] += 1
                    for k in ov: seen.add(k); found_by[k] += 1
                elif action == "MASK" and label in COMMON:
                    rr["review_only"] += 1  # レビューだけが MASK と指摘（見落とし候補）
                    added_candidates[(s, e, label)] = added_candidates.get((s, e, label), 0) + 1
                else:
                    rr["keep_only"] += 1    # KEEP/REVIEW の指摘で正解に実体がない（会社名・型番等）。不一致には数えない
        rr["missed_mask_gold"] = len([k for k in mask_gold if k not in seen])
        if rr["boundary_mismatch"] or rr["label_mismatch"] or rr["action_mismatch"] or rr["missed_mask_gold"] or rr["review_only"]:
            result["agree_all"] = False
        result["per_review"].append(rr)
    result["missed_by_reviewers"] = [list(k) for k in mask_gold if found_by[k] == 0]
    result["reviewer_added"] = [{"start": s, "end": e, "label": l, "text": text[s:e], "votes": v} for (s, e, l), v in added_candidates.items()]
    n_ok = sum(1 for r in result["per_review"] if r["json_ok"] and not r["unreviewable"])
    if n_ok < 2:
        result["review_status"] = "unreviewed"
    elif result["agree_all"]:
        result["review_status"] = "auto_reviewed"
    else:
        result["review_status"] = "auto_reviewed_disagreement"
    return result


def apply_reconciliation(doc: dict, rec: dict) -> dict:
    """文書へ review_status と、両レビュー一致の追加候補（REVIEW として）を反映する。"""
    d = dict(doc)
    d["review_status"] = rec["review_status"]
    ents = list(d.get("entities", []))
    n_rev = max(1, sum(1 for r in rec["per_review"] if r["json_ok"] and not r["unreviewable"]))
    for c in rec["reviewer_added"]:
        if c["votes"] >= n_rev and not any(e["start"] < c["end"] and c["start"] < e["end"] for e in ents):
            ents.append({"start": c["start"], "end": c["end"], "text": c["text"], "label": c["label"], "action": "REVIEW",
                         "entity_id": f"rev-{c['start']}", "source": "reviewer_added", "note": f"両レビューが MASK 指摘（votes={c['votes']}）。人手確認待ち"})
    ents.sort(key=lambda e: e["start"])
    probs = validate_spans(d["text"], [Span(e["start"], e["end"], e["label"], e["text"], e["action"]) for e in ents])
    if probs:
        d["review_status"] = "quarantined"; d.setdefault("provenance", {})["notes"] = (d.get("provenance", {}).get("notes", "") + " | reconcile problems: " + "; ".join(probs[:3])).strip(" |")
    d["entities"] = ents
    d["reviews"] = [{"reviewer_model": r["reviewer"], "reviewed_at": "", "verdict": ("cannot_review" if r["unreviewable"] else ("agree" if not (r["boundary_mismatch"] or r["label_mismatch"] or r["action_mismatch"] or r["missed_mask_gold"] or r["review_only"]) else "disagree"))} for r in rec["per_review"]]
    return d


def reconcile_files(docs_path: str, reviews_dir: str, out_docs: str, out_report: str) -> dict:
    docs = [json.loads(l) for l in open(docs_path, encoding="utf-8") if l.strip()]
    rdir = Path(reviews_dir)
    stats = {"docs": len(docs), "auto_reviewed": 0, "auto_reviewed_disagreement": 0, "unreviewed": 0, "quarantined": 0, "reviewer_added_candidates": 0, "missed_by_reviewers": 0}
    recs = []
    with open(out_docs, "w", encoding="utf-8") as f:
        for d in docs:
            reviews = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(rdir.glob(f"{d['id']}.*.json"))]
            rec = reconcile_doc(d, reviews)
            nd = apply_reconciliation(d, rec)
            stats[nd["review_status"]] += 1
            stats["reviewer_added_candidates"] += len(rec["reviewer_added"]); stats["missed_by_reviewers"] += len(rec["missed_by_reviewers"])
            recs.append(rec)
            f.write(json.dumps(nd, ensure_ascii=False) + "\n")
    Path(out_report).write_text(json.dumps({"stats": stats, "docs": recs}, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True); ap.add_argument("--reviews", required=True); ap.add_argument("--out-docs", required=True); ap.add_argument("--out-report", required=True)
    a = ap.parse_args()
    print(json.dumps(reconcile_files(a.docs, a.reviews, a.out_docs, a.out_report), ensure_ascii=False))
