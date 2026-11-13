#!/usr/bin/env python3
"""precision / recall 曲線。閾値 0 で出した予測（span に score あり）を事後に閾値で絞り、strict micro P/R/F1 と運用指標を閾値ごとに集計する。
  .venv/bin/python scripts/pr_curve.py --gold eval/locked/test.jsonl --preds T-O=runs/final/pr/T-O_thr0.pred.jsonl T-S=... S=... --out runs/final/pr
出力: pr_curves.csv / pr_curves.md / pr_curves.svg（matplotlib 不要）
"""
import argparse, csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toramon.evaluate import load_gold_jsonl, load_pred_jsonl, score_doc, aggregate
ap = argparse.ArgumentParser(); ap.add_argument("--gold", required=True); ap.add_argument("--preds", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--thresholds", default="0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98")
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
gold = load_gold_jsonl(a.gold); ths = [float(x) for x in a.thresholds.split(",")]
rows = []
for kv in a.preds:
    name, p = kv.split("=", 1); pr = load_pred_jsonl(p)
    for t in ths:
        for d in gold: d.pred = [s for s in pr.get(d.id, []) if (s.score if s.score is not None else 1.0) >= t]
        ag = aggregate([score_doc(d) for d in gold]); m = ag["micro"]
        rows.append({"model": name, "threshold": t, "precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "full_coverage_rate": ag["full_coverage_rate"],
                     "pii_remaining_doc_rate": ag["pii_remaining_doc_rate"], "nontarget_char_mask_rate": ag["nontarget_char_mask_rate"], "negative_doc_fp_rate": ag["negative_doc_fp_rate"]})
with open(out / "pr_curves.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
md = [f"# precision / recall 曲線（{a.gold}、モデル単体、閾値は span の平均確率）", "", "| モデル | 閾値 | P | R | strict F1 | 全範囲保護率 | PII残存文書率 | 非対象文字マスク率 | 負例誤検知率 |", "|---|---|---|---|---|---|---|---|---|"]
for r in rows: md.append(f"| {r['model']} | {r['threshold']:.2f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['full_coverage_rate']*100:.1f}% | {r['pii_remaining_doc_rate']*100:.1f}% | {r['nontarget_char_mask_rate']*100:.2f}% | {r['negative_doc_fp_rate']*100:.1f}% |")
best = {}
for r in rows:
    if r["model"] not in best or r["f1"] > best[r["model"]]["f1"]: best[r["model"]] = r
md += ["", "F1 最大の閾値: " + "、".join(f"{k} {v['threshold']:.2f}（F1 {v['f1']:.3f}）" for k, v in best.items()), "", "![PR](pr_curves.svg)"]
(out / "pr_curves.md").write_text("\n".join(md) + "\n", encoding="utf-8")
# SVG（P を x、R を y）
W, H, pad = 520, 520, 60; colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
def X(p): return pad + p * (W - 2 * pad)
def Y(r): return H - pad - r * (H - 2 * pad)
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="12">', f'<rect width="{W}" height="{H}" fill="white"/>']
for i in range(0, 11):
    v = i / 10; svg.append(f'<line x1="{X(v)}" y1="{Y(0)}" x2="{X(v)}" y2="{Y(1)}" stroke="#eee"/>'); svg.append(f'<line x1="{X(0)}" y1="{Y(v)}" x2="{X(1)}" y2="{Y(v)}" stroke="#eee"/>')
    svg.append(f'<text x="{X(v)}" y="{Y(0)+16}" text-anchor="middle">{v:.1f}</text>'); svg.append(f'<text x="{X(0)-8}" y="{Y(v)+4}" text-anchor="end">{v:.1f}</text>')
svg.append(f'<text x="{W/2}" y="{H-12}" text-anchor="middle">precision</text>'); svg.append(f'<text x="14" y="{H/2}" text-anchor="middle" transform="rotate(-90 14 {H/2})">recall</text>')
for ci, (name, _) in enumerate([kv.split("=", 1) for kv in a.preds]):
    pts = [(r["precision"], r["recall"], r["threshold"]) for r in rows if r["model"] == name]
    svg.append(f'<polyline fill="none" stroke="{colors[ci % 5]}" stroke-width="2" points="' + " ".join(f"{X(p):.1f},{Y(r):.1f}" for p, r, _ in pts) + '"/>')
    for p, r, t in pts:
        svg.append(f'<circle cx="{X(p):.1f}" cy="{Y(r):.1f}" r="{4 if abs(t-0.5)<1e-9 else 2.5}" fill="{colors[ci % 5]}"/>')
    svg.append(f'<text x="{X(0.02)}" y="{Y(0.98 - 0.05*ci)}" fill="{colors[ci % 5]}">{name}（● は閾値 0.5）</text>')
svg.append("</svg>"); (out / "pr_curves.svg").write_text("\n".join(svg), encoding="utf-8")
print("\n".join(md[:4] + [l for l in md if "| 0.50 |" in l or "| 0.30 |" in l or "| 0.70 |" in l] + md[-3:-1]))
