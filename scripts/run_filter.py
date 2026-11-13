#!/usr/bin/env python3
"""モデル + 共通ルール R（フィルタ構成）で予測 JSONL を出す。ルール ON/OFF・境界補正のみ・検出ルールのみを切り替えて比較できる。

  .venv/bin/python scripts/run_filter.py --backend S --gguf <f16> --input gold.jsonl --out pred.jsonl --config rules_all
  configs: model_only | boundary_only | detect_only | rules_all | no_suppress
"""
from __future__ import annotations
import argparse, json, sys, time, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toramon.filter import ToramonFilter, Policy  # noqa: E402

CONFIGS = {
    "model_only": dict(use_rules=False, use_boundary_fix=False, use_keep_suppression=False),
    "boundary_only": dict(use_rules=False, use_boundary_fix=True, use_keep_suppression=False),
    "detect_only": dict(use_rules=True, use_boundary_fix=False, use_keep_suppression=False),
    "no_suppress": dict(use_rules=True, use_boundary_fix=True, use_keep_suppression=False),
    "rules_all": dict(use_rules=True, use_boundary_fix=True, use_keep_suppression=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["S", "O", "HF"]); ap.add_argument("--gguf"); ap.add_argument("--checkpoint"); ap.add_argument("--label-space", default="identity")
    ap.add_argument("--input", required=True); ap.add_argument("--out", required=True); ap.add_argument("--config", default="rules_all", choices=list(CONFIGS))
    ap.add_argument("--device", default="cuda"); ap.add_argument("--threshold", type=float, default=0.5); ap.add_argument("--rules", default=None); ap.add_argument("--o-bias", type=float, default=0.0, help="HF 復号器の O ラベル対数尤度バイアス（>0 で再現率寄り）")
    ap.add_argument("--decode", default="libpf", choices=["libpf", "toramon"], help="backend S: libpf の C 復号器か、生ロジットを Toramon 復号器（HF adapter と同一）で復号するか")
    a = ap.parse_args()
    if a.backend == "S":
        if a.decode == "toramon":
            from toramon.adapters.sumeshi_pf import SumeshiPFDecoded
            det = SumeshiPFDecoded(a.gguf, device=a.device, threshold=a.threshold, label_space=a.label_space, o_bias=a.o_bias)
            mv = f"GGUF-toramon:{Path(a.gguf).name}:{a.label_space}" + (f"@obias{a.o_bias}" if a.o_bias else "")
        else:
            from toramon.adapters.sumeshi_pf import SumeshiPF
            det = SumeshiPF(a.gguf, device=a.device, threshold=a.threshold); mv = f"S:{Path(a.gguf).name}"
    elif a.backend == "O":
        from toramon.adapters.openai_opf import OpenAIOPF
        det = OpenAIOPF(a.checkpoint, device=a.device); mv = f"O:{a.checkpoint}"
    else:
        from toramon.adapters.hf_token_cls import HFTokenClassifier
        det = HFTokenClassifier(a.checkpoint, device=a.device, label_space=a.label_space, threshold=a.threshold, o_bias=a.o_bias); mv = f"HF:{a.checkpoint}" + (f"@obias{a.o_bias}" if a.o_bias else "")
    f = ToramonFilter(det, mv, rules_path=a.rules, policy=Policy(**CONFIGS[a.config]))
    rows = [json.loads(l) for l in open(a.input, encoding="utf-8") if l.strip()]
    lat = []; fails = 0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fo:
        for r in rows:
            res = f.detect(r["text"])
            lat.append(res.elapsed_ms)
            if res.status != "ok":
                fails += 1
                fo.write(json.dumps({"id": r["id"], "spans": [], "error": res.error}, ensure_ascii=False) + "\n"); continue
            fo.write(json.dumps({"id": r["id"], "spans": [{"start": s["start"], "end": s["end"], "label": s["label"], "text": r["text"][s["start"]:s["end"]], "score": s["score"], "source": s["source"]} for s in res.spans], "elapsed_ms": res.elapsed_ms}, ensure_ascii=False) + "\n")
    meta = {"backend": mv, "config": a.config, "rules_version": f.rules_version, "policy": CONFIGS[a.config], "n": len(rows), "failures": fails,
            "p50_ms": statistics.median(lat), "p95_ms": sorted(lat)[int(0.95 * (len(lat) - 1))]}
    Path(a.out + ".run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
