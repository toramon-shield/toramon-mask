#!/usr/bin/env python3
"""O バイアス掃引（コンテナ内）。logprobs を 1 回計算し、O の対数尤度から bias を引いて Viterbi を再実行、dev 指標を算出。
  env/run.sh python scripts/sweep_obias.py --checkpoint /work/runs/exp_real_v5/TO_ef_lr1e-4_e3_pw2/model --label-space identity --gold /work/data/dev/dev_v1_749.jsonl --biases 0,0.5,1,1.5,2,3 --out /work/runs/sweeps/TO_v5_obias.json
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
    ap.add_argument("--biases", default="0,0.5,1,1.5,2,3"); ap.add_argument("--threshold", type=float, default=0.5); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    clf = HFTokenClassifier(a.checkpoint, device="cuda", label_space=a.label_space, threshold=a.threshold)
    docs = load_gold_jsonl(a.gold)
    cache = []; t0 = time.perf_counter()
    for d in docs:
        lp, offs = clf.token_logprobs(d.text); cache.append((d, lp, offs))
    print("inference s", round(time.perf_counter() - t0, 1), flush=True)
    results = {}
    for b in [float(x) for x in a.biases.split(",")]:
        counts = []
        for d, lp, offs in cache:
            spans = []
            if lp.shape[0] > 0:
                lp2 = lp.clone(); lp2[:, clf.tb.o_id] -= b
                path = viterbi_bioes(lp2, clf.tb); probs = lp.exp()
                for tb_, te_, cat, score in spans_from_path(path, probs, clf.tb):
                    if score < a.threshold: continue
                    label = cat if clf.label_map is None else clf.label_map.get(cat)
                    if label is None: continue
                    cs, ce = offs[tb_][0], offs[te_][1]; text = d.text
                    while cs < ce and (text[cs].isspace() or text[cs] == "="): cs += 1
                    while ce > cs and text[ce - 1].isspace(): ce -= 1
                    if cs < ce: spans.append(Span(cs, ce, label, text[cs:ce], "MASK", "", score))
            d.pred = sorted(spans, key=lambda s: (s.start, s.end)); counts.append(score_doc(d))
        r = aggregate(counts)
        row = {"micro_f1": round(r["micro"]["f1"], 4), "precision": round(r["micro"]["precision"], 4), "recall": round(r["micro"]["recall"], 4),
               "full_coverage": round(r["full_coverage_rate"], 4), "pii_remaining": round(r["pii_remaining_doc_rate"], 4),
               "nontarget_mask": round(r["nontarget_char_mask_rate"], 4), "neg_fp": round(r["negative_doc_fp_rate"], 4),
               "important_cov": {k: round(r["per_label"][k]["full_coverage_rate"], 3) for k in ("private_person", "private_address") if k in r["per_label"]}}
        results[str(b)] = row; print(b, row, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"checkpoint": a.checkpoint, "gold": a.gold, "threshold": a.threshold, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")

if __name__ == "__main__":
    main()
