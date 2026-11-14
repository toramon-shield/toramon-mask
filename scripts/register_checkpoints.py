#!/usr/bin/env python3
"""比較用の公開モデル S（sumeshi 単体）と比べて全指標で良かった学習チェックポイントを checkpoints/ に登録・保全する。
  .venv/bin/python scripts/register_checkpoints.py [--gold data/dev/dev_v1_749.jsonl] [--baseline runs/dev_compare_v3/S_model_only.eval.json]
- runs/exp_real_v*/<run>/dev_pred.jsonl（学習スクリプト内評価、閾値 0.5、単体）を gold で再採点し、S と比較。
- 比較指標: strict micro F1（高）、全範囲保護率（高）、PII 残存文書率（低）、非対象文字マスク率（低）、負例誤検知率（低）。全て上回った run のみ登録。
- 登録: checkpoints/<version>__<run> → runs/.../model へのシンボリックリンク、checkpoints/registry.jsonl に指標と SHA を追記（重複登録はしない）。
- 登録済みチェックポイントは削除・上書きしない（学習スクリプトは別名で保存する）。
"""
import argparse, json, glob, hashlib, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from toramon.evaluate import load_gold_jsonl, score_doc, aggregate
from toramon.spans import Span

def metrics_from_pred(gold_path, pred_path):
    docs = {d.id: d for d in load_gold_jsonl(gold_path)}
    n = 0
    for l in open(pred_path, encoding="utf-8"):
        r = json.loads(l); d = docs.get(r.get("id") or r.get("doc_id"))
        if d is None: continue
        d.pred = [Span(s["start"], s["end"], s["label"], s.get("text", d.text[s["start"]:s["end"]]), "MASK", "", s.get("score")) for s in r["spans"]]; n += 1
    if n < len(docs) * 0.95: return None, n
    a = aggregate([score_doc(d) for d in docs.values()])
    return {"micro_f1": a["micro"]["f1"], "full_coverage_rate": a["full_coverage_rate"], "pii_remaining_doc_rate": a["pii_remaining_doc_rate"],
            "nontarget_char_mask_rate": a["nontarget_char_mask_rate"], "negative_doc_fp_rate": a["negative_doc_fp_rate"]}, n

def surpasses(m, b):
    return (m["micro_f1"] > b["micro_f1"] and m["full_coverage_rate"] > b["full_coverage_rate"] and m["pii_remaining_doc_rate"] < b["pii_remaining_doc_rate"]
            and m["nontarget_char_mask_rate"] < b["nontarget_char_mask_rate"] and m["negative_doc_fp_rate"] < b["negative_doc_fp_rate"])

def model_sha(model_dir):
    h = hashlib.sha256()
    for p in sorted(Path(model_dir).glob("*.safetensors")):
        h.update(p.name.encode()); h.update(str(p.stat().st_size).encode())
    return h.hexdigest()[:16]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gold", default=str(ROOT / "data/dev/dev_v1_749.jsonl"))
    ap.add_argument("--baseline", default=str(ROOT / "runs/dev_compare_v3/S_model_only.eval.json")); ap.add_argument("--runs-glob", default="runs/exp_real_v*/*")
    a = ap.parse_args()
    b = json.load(open(a.baseline, encoding="utf-8")); base = {"micro_f1": b["micro"]["f1"], "full_coverage_rate": b["full_coverage_rate"], "pii_remaining_doc_rate": b["pii_remaining_doc_rate"],
                                                               "nontarget_char_mask_rate": b["nontarget_char_mask_rate"], "negative_doc_fp_rate": b["negative_doc_fp_rate"]}
    reg = ROOT / "checkpoints/registry.jsonl"; reg.parent.mkdir(exist_ok=True)
    done = {json.loads(l)["key"] for l in open(reg, encoding="utf-8")} if reg.exists() else set()
    rows = []
    for run in sorted(glob.glob(str(ROOT / a.runs_glob))):
        run = Path(run); pred = run / "dev_pred.jsonl"; model = run / "model"
        if not pred.exists() or not (model / "config.json").exists(): continue
        key = f"{run.parent.name}__{run.name}"
        m, n = metrics_from_pred(a.gold, pred)
        if m is None: rows.append((key, "skip: gold mismatch", n)); continue
        ok = surpasses(m, base)
        rows.append((key, "SURPASSES_S" if ok else "below", {k: round(v, 4) for k, v in m.items()}))
        if ok and key not in done:
            link = ROOT / "checkpoints" / key
            if not link.exists(): os.symlink(os.path.relpath(model, link.parent), link)
            with open(reg, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "model_dir": str(model), "model_sha": model_sha(model), "metrics": m, "baseline": base, "gold": a.gold,
                                    "registered_at": __import__("datetime").datetime.now().astimezone().isoformat()}, ensure_ascii=False) + "\n")
            done.add(key)
    for r in rows: print(*r)
    print("registered:", len(done))

if __name__ == "__main__":
    main()
