#!/usr/bin/env python3
"""評価結果（*.eval.json / *.run.json / 比較 JSON）から日本語の評価報告 Markdown と集計 CSV を生成する。

  python scripts/make_report.py --title "dev 基準値" --evals runs/dev/*.eval.json --out runs/reports/dev_baseline.md [--compare runs/dev/compare_*.json]
各 eval JSON は scripts/evaluate.py --out の出力。ファイル名（例 S_f16_model_only_dev.eval.json）を条件名として使う。
"""
from __future__ import annotations
import argparse, csv, json, datetime
from pathlib import Path


def pct(x, nd=1):
    return "N/A" if x is None else f"{100*x:.{nd}f}%"


def f3(x):
    return "N/A" if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--title", required=True); ap.add_argument("--evals", nargs="+", required=True)
    ap.add_argument("--compare", nargs="*", default=[]); ap.add_argument("--out", required=True); ap.add_argument("--note", default="")
    ap.add_argument("--samples", default=None, help="make_samples.py の出力 Markdown を末尾に取り込む")
    ap.add_argument("--promo", default=None, help="訴求（優位性）節: 候補名=基準名 を指定すると自動生成（例 T-O-ef_model_only=S_model_only）")
    a = ap.parse_args()
    rows = []
    for p in a.evals:
        e = json.load(open(p, encoding="utf-8")); name = Path(p).name.replace(".eval.json", "")
        run = Path(p.replace(".eval.json", ".pred.jsonl.run.json"))
        r = json.load(open(run, encoding="utf-8")) if run.exists() else {}
        ci = e.get("ci95", {})
        rows.append({"condition": name, "n_docs": e["n_docs"], "n_pii_docs": e["n_docs_with_pii"], "n_neg": e["n_docs_negative"],
                     "micro_f1": e["micro"]["f1"], "micro_f1_ci": ci.get("micro_f1", {}).get("ci95"), "macro_f1": e["macro_f1"],
                     "precision": e["micro"]["precision"], "recall": e["micro"]["recall"], "full_coverage": e["full_coverage_rate"],
                     "full_coverage_ci": ci.get("full_coverage_rate", {}).get("ci95"), "pii_remaining": e["pii_remaining_doc_rate"],
                     "pii_remaining_ci": ci.get("pii_remaining_doc_rate", {}).get("ci95"), "nontarget_mask": e["nontarget_char_mask_rate"],
                     "neg_fp": e["negative_doc_fp_rate"], "failures": e.get("missing_predictions", 0),
                     "p50_ms": (r.get("latency_ms") or {}).get("p50") or r.get("p50_ms"), "p95_ms": (r.get("latency_ms") or {}).get("p95") or r.get("p95_ms"),
                     "per_label": e["per_label"], "breakdown": e.get("breakdown", {})})
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    L = [f"# {a.title}", "", f"生成日時: {datetime.datetime.now().astimezone().isoformat(timespec='minutes')}", ""]
    if a.note: L += [a.note, ""]
    L += ["## 主要指標", "", "| 条件 | 文書数(PIIあり/負例) | strict micro F1 [95%CI] | macro F1 | P | R | 全範囲保護率 [CI] | PII残存文書率 [CI] | 非対象文字マスク率 | 負例誤検知率 | 失敗 | p50/p95 ms |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    def ci(c): return "" if not c or c[0] is None else f" [{f3(c[0])}, {f3(c[1])}]"
    def cip(c): return "" if not c or c[0] is None else f" [{pct(c[0])}, {pct(c[1])}]"
    for r in rows:
        L.append(f"| {r['condition']} | {r['n_docs']} ({r['n_pii_docs']}/{r['n_neg']}) | {f3(r['micro_f1'])}{ci(r['micro_f1_ci'])} | {f3(r['macro_f1'])} | {f3(r['precision'])} | {f3(r['recall'])} | {pct(r['full_coverage'])}{cip(r['full_coverage_ci'])} | {pct(r['pii_remaining'])}{cip(r['pii_remaining_ci'])} | {pct(r['nontarget_mask'],2)} | {pct(r['neg_fp'])} | {r['failures']} | {r['p50_ms'] if r['p50_ms'] is None else round(r['p50_ms'])}/{r['p95_ms'] if r['p95_ms'] is None else round(r['p95_ms'])} |")
    L += ["", "## カテゴリ別（precision / recall / F1 / 全範囲保護率）", ""]
    labels = ["private_person", "private_address", "private_email", "private_phone", "private_date", "account_number", "private_url", "secret"]
    L.append("| 条件 | " + " | ".join(labels) + " |"); L.append("|---|" + "---|" * len(labels))
    for r in rows:
        cells = []
        for lb in labels:
            v = r["per_label"].get(lb, {})
            cells.append(f"{f3(v.get('precision'))}/{f3(v.get('recall'))}/{f3(v.get('f1'))}/{pct(v.get('full_coverage_rate'),0)} (n={v.get('gold_spans',0)})")
        L.append(f"| {r['condition']} | " + " | ".join(cells) + " |")
    for key in ("domain", "generator_model", "length_band", "difficulty", "negative_type"):
        if any(key in r["breakdown"] for r in rows):
            L += ["", f"## 内訳: {key}（micro F1 / 全範囲保護率 / PII残存文書率、文書数）", ""]
            groups = sorted({g for r in rows for g in r["breakdown"].get(key, {})})
            L.append("| 条件 | " + " | ".join(groups) + " |"); L.append("|---|" + "---|" * len(groups))
            for r in rows:
                cells = []
                for g in groups:
                    v = r["breakdown"].get(key, {}).get(g)
                    cells.append("-" if not v else f"{f3(v['micro']['f1'])} / {pct(v['full_coverage_rate'],0)} / {pct(v['pii_remaining_doc_rate'],0)} (n={v['n_docs']})")
                L.append(f"| {r['condition']} | " + " | ".join(cells) + " |")
    if a.compare:
        L += ["", "## 対応付き比較（A − B、文書/シナリオ群単位 bootstrap 95%CI）", "", "| 比較 | 指標 | A | B | 差 | 95%CI |", "|---|---|---|---|---|---|"]
        for p in a.compare:
            c = json.load(open(p, encoding="utf-8")); name = Path(p).stem
            for metric, v in c.items():
                if not isinstance(v, dict) or "ci95" not in v: continue
                L.append(f"| {name} | {metric} | {f3(v.get('point_a'))} | {f3(v.get('point_b'))} | {f3(v.get('point_diff'))} | [{f3(v['ci95'][0])}, {f3(v['ci95'][1])}] |")
    if a.promo:
        cand_name, base_name = a.promo.split("=", 1)
        byname = {r["condition"]: r for r in rows}
        c, b = byname.get(cand_name), byname.get(base_name)
        if c and b:
            def d(k, pct_=False, lower_better=False):
                cv, bv = c[k], b[k]
                if cv is None or bv is None: return "N/A"
                diff = (cv - bv) * (100 if pct_ else 1)
                better = (diff < 0) if lower_better else (diff > 0)
                arrow = "改善" if better else "悪化"
                return f"{(cv*100 if pct_ else cv):.{1 if pct_ else 3}f}{'%' if pct_ else ''} vs {(bv*100 if pct_ else bv):.{1 if pct_ else 3}f}{'%' if pct_ else ''}（{diff:+.{1 if pct_ else 3}f}{'pt' if pct_ else ''}、{arrow}）"
            wins = []; losses = []
            for k, label, pct_, lb in [("micro_f1", "strict span F1（検出範囲の精密さ）", False, False), ("precision", "適合率（余計に伏せない）", False, False), ("recall", "再現率（取りこぼさない）", False, False),
                                       ("full_coverage", "全範囲保護率（伏せ残しの少なさ）", True, False), ("pii_remaining", "PII 残存文書率", True, True),
                                       ("nontarget_mask", "非対象文字マスク率（過剰マスキング）", True, True), ("neg_fp", "負例文書誤検知率（PII のない文書を壊さない）", True, True)]:
                cv, bv = c[k], b[k]
                if cv is None or bv is None: continue
                better = (cv < bv) if lb else (cv > bv)
                (wins if better else losses).append(f"- **{label}**: {d(k, pct_, lb)}")
            L += ["", f"## 学習版（{cand_name}）と基準 {base_name} の比較", ""]
            L += ["### 上回っている点", ""] + (wins or ["- （なし）"]) + ["", "### 下回っている点（次の学習ラウンドで解消する対象）", ""] + (losses or ["- （なし）"]) + [""]
            L += ["### 位置づけ（事実に基づく表現）", "",
                  f"- 本比較は dev {c['n_docs']} 文書（合成・Toramon 保護方針・暫定正解）での結果であり、実務文書への一般化は主張しない。",
                  "- 元モデルの検出方針は「日付や企業連絡先も広く伏せる」ため、業務文書では残すべき情報まで消えてしまう。Toramon 学習版は日本の業務文脈で「個人に紐づく情報だけを伏せ、業務上必要な情報（代表番号・共用メール・納期）は残す」方針を学習している。",
                  "- 見逃し側（全範囲保護率）で基準を下回る場合は『モデル単体で上回った』とは表現せず、共通ルール R 併用の数値と次ラウンドの改善計画を併記する。", ""]
    L += ["", "## 注記", "", "- strict F1 は開始・終了・ラベルの完全一致。全範囲保護率は正解 MASK span の全文字がマスク範囲に含まれる割合（ラベル違いも含む）。", "- PII 残存文書率の分母は PII あり文書。空分母は N/A。", "- 公開ベンチの数値は診断用（作者方針の正解）であり、Toramon 方針の dev/test とは比較しない。"]
    if a.samples and Path(a.samples).exists():
        L += ["", Path(a.samples).read_text(encoding="utf-8").replace("# マスク前／マスク後サンプル", "## マスク前／マスク後サンプル")]
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["condition", "n_docs", "micro_f1", "macro_f1", "precision", "recall", "full_coverage", "pii_remaining", "nontarget_mask", "neg_fp", "failures", "p50_ms", "p95_ms"])
        for r in rows: w.writerow([r["condition"], r["n_docs"], r["micro_f1"], r["macro_f1"], r["precision"], r["recall"], r["full_coverage"], r["pii_remaining"], r["nontarget_mask"], r["neg_fp"], r["failures"], r["p50_ms"], r["p95_ms"]])
    print("wrote", out, out.with_suffix(".csv"))


if __name__ == "__main__":
    main()
