#!/usr/bin/env python3
"""HF 形式モデルの運用点探索（コンテナ内で実行）。モデルを 1 回ロードし、閾値を変えて dev 指標を一括計算する。
  env/run.sh python scripts/sweep_threshold.py --checkpoint /work/runs/exp_real_v1/TO_ef_lr1e-4_e3/model --label-space identity --gold /work/data/dev/reconciled_prelim.jsonl --out /work/runs/sweeps/TO_ef_v1.json
選定基準（acceptance.yaml）: 全範囲保護率が基準 S を 0.5pt 以上下回らない範囲で strict F1 最大。
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, "/work/src" if Path("/work/src").exists() else str(Path(__file__).resolve().parents[1] / "src"))
import torch
from toramon.adapters.hf_token_cls import HFTokenClassifier, viterbi_bioes, spans_from_path
from toramon.spans import Span
from toramon.evaluate import load_gold_jsonl, score_doc, aggregate


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", required=True); ap.add_argument("--label-space", default="identity"); ap.add_argument("--gold", required=True)
    ap.add_argument("--thresholds", default="0.5,0.4,0.3,0.2,0.1,0.05"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    clf = HFTokenClassifier(a.checkpoint, device="cuda", label_space=a.label_space, threshold=0.0)
    docs = load_gold_jsonl(a.gold)
    # logits を 1 回だけ計算し、閾値ごとに span を再構成
    cache = []
    t0 = time.perf_counter()
    for d in docs:
        lp, offs = clf.token_logprobs(d.text)
        if lp.shape[0] == 0:
            cache.append((d, [], offs)); continue
        path = viterbi_bioes(lp, clf.tb); probs = lp.exp()
        cache.append((d, spans_from_path(path, probs, clf.tb), offs))
    print("inference s", round(time.perf_counter() - t0, 1), flush=True)
    results = {}
    for thr in [float(x) for x in a.thresholds.split(",")]:
        counts = []
        for d, raw, offs in cache:
            spans = []
            for tb_, te_, cat, score in raw:
                if score < thr: continue
                label = cat if clf.label_map is None else clf.label_map.get(cat)
                if label is None: continue
                cs, ce = offs[tb_][0], offs[te_][1]; text = d.text
                while cs < ce and (text[cs].isspace() or text[cs] == "="): cs += 1
                while ce > cs and text[ce - 1].isspace(): ce -= 1
                if cs < ce: spans.append(Span(cs, ce, label, text[cs:ce], "MASK", "", score))
            d.pred = sorted(spans, key=lambda s: (s.start, s.end)); counts.append(score_doc(d))
        r = aggregate(counts)
        results[str(thr)] = {"micro_f1": r["micro"]["f1"], "precision": r["micro"]["precision"], "recall": r["micro"]["recall"], "full_coverage": r["full_coverage_rate"],
                             "pii_remaining": r["pii_remaining_doc_rate"], "nontarget_mask": r["nontarget_char_mask_rate"], "neg_fp": r["negative_doc_fp_rate"]}
        print(thr, {k: round(v, 4) for k, v in results[str(thr)].items()}, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps({"checkpoint": a.checkpoint, "gold": a.gold, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
