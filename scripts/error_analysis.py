#!/usr/bin/env python3
"""残存エラーの分類。FN/FP を次の種別に自動分類し、件数と例を出す。

  形式の見逃し(format_miss: 正規表現で捕まる形式の FN) / 未知の人名・住所(unknown_name_address: person/address の完全 FN) /
  境界ずれ(boundary: 重なりはあるが不一致) / 意味・方針判断(policy: KEEP 領域のマスク、または日付・URL 等の方針差) /
  過検知(over_detection: 正解と重ならない FP) / 文字コード・チャンク境界(encoding_chunk: 全角半角・結合文字・長文由来) / その他
  python scripts/error_analysis.py --gold gold.jsonl --pred pred.jsonl --out runs/.../errors.json [--examples 5]
"""
from __future__ import annotations
import argparse, json, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toramon.evaluate import load_gold_jsonl, load_pred_jsonl  # noqa: E402
from toramon.rules import RuleEngine  # noqa: E402


def has_width_mix(s: str) -> bool:
    kinds = {unicodedata.east_asian_width(c) for c in s if c.isalnum()}
    return len(kinds & {"F", "Na", "H", "W"}) > 1 and ("F" in kinds or "H" in kinds)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gold", required=True); ap.add_argument("--pred", required=True); ap.add_argument("--out"); ap.add_argument("--examples", type=int, default=5)
    a = ap.parse_args()
    docs = load_gold_jsonl(a.gold); preds = load_pred_jsonl(a.pred)
    eng = RuleEngine()
    cats: dict[str, list] = defaultdict(list)
    by_label = defaultdict(Counter)
    for d in docs:
        pred = preds.get(d.id, [])
        gold_mask = [g for g in d.gold if g.action == "MASK"]
        gold_keep = [g for g in d.gold if g.action == "KEEP"]
        pkeys = {p.key() for p in pred}; gkeys = {g.key() for g in gold_mask}
        rule_hits = {(s.start, s.end) for s in eng.detect(d.text)}
        for g in gold_mask:
            if g.key() in pkeys: continue
            ov = [p for p in pred if p.overlaps(g)]
            if ov:
                kind = "boundary" if any(p.label == g.label for p in ov) else "label_confusion"
            elif (g.start, g.end) in rule_hits or any(rs <= g.start and g.end <= re_ for rs, re_ in rule_hits):
                kind = "format_miss"
            elif g.label in ("private_person", "private_address"):
                kind = "unknown_name_address"
            elif has_width_mix(g.text) or any(unicodedata.combining(c) for c in g.text) or len(d.text) > 4000:
                kind = "encoding_chunk"
            else:
                kind = "semantic_miss"
            cats[f"FN:{kind}"].append({"id": d.id, "label": g.label, "text": g.text, "pred_overlap": [(p.label, p.text) for p in ov][:2]})
            by_label[g.label][f"FN:{kind}"] += 1
        for p in pred:
            if p.key() in gkeys: continue
            if any(p.overlaps(g) for g in gold_mask): continue  # 境界ずれは FN 側で数える
            if any(p.overlaps(k) for k in gold_keep):
                kind = "policy_keep_masked"
            elif p.label in ("private_date", "private_url"):
                kind = "policy_date_url"
            else:
                kind = "over_detection"
            cats[f"FP:{kind}"].append({"id": d.id, "label": p.label, "text": p.text, "source": p.source})
            by_label[p.label][f"FP:{kind}"] += 1
    summary = {k: len(v) for k, v in sorted(cats.items(), key=lambda kv: -len(kv[1]))}
    out = {"summary": summary, "by_label": {k: dict(v) for k, v in by_label.items()}, "examples": {k: v[: a.examples] for k, v in cats.items()},
           "guidance": {"format_miss": "正規表現・形式検証で補完", "unknown_name_address": "追加学習を優先", "boundary": "境界補正と追加学習を比較",
                        "policy_*": "保護方針の差。ルール（KEEP 抑制）または方針データで学習", "over_detection": "負例追加・閾値", "encoding_chunk": "正規化・チャンク境界の処理"}}
    js = json.dumps(out, ensure_ascii=False, indent=1)
    if a.out: Path(a.out).write_text(js, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
