#!/usr/bin/env python3
"""共通入力（Toramon JSONL）に対してモデルを実行し、予測 JSONL と速度・失敗を記録する。

  python scripts/run_model.py --model S --gguf <path> --input gold.jsonl --out pred.jsonl [--device cuda] [--threshold 0.5]
  python scripts/run_model.py --model O --checkpoint <dir> --input gold.jsonl --out pred.jsonl [--decode viterbi|argmax]

出力 JSONL: {"id", "spans":[{start,end,label,text,score?,source}], "elapsed_ms", "error"?}
サイドカー: <out>.run.json に設定・所要時間・失敗数・環境を記録。
"""
from __future__ import annotations

import argparse, json, os, platform, statistics, sys, time, datetime, hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toramon.spans import Span  # noqa: E402


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["S", "O", "T-O", "T-S"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gguf")
    ap.add_argument("--checkpoint")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--decode", default="viterbi", choices=["viterbi", "argmax"])
    ap.add_argument("--no-merge", action="store_true", help="S: 同カテゴリ隣接 span の結合を無効化")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.input, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[: a.limit]

    cfg: dict = {"model": a.model, "device": a.device, "input": a.input, "n_docs": len(rows), "tag": a.tag,
                 "host": platform.node(), "started_at": datetime.datetime.now().astimezone().isoformat()}
    if a.model in ("S", "T-S"):
        from toramon.adapters.sumeshi_pf import SumeshiPF
        m = SumeshiPF(a.gguf, device=a.device, threshold=a.threshold)
        cfg.update(gguf=a.gguf, gguf_sha256=sha256_file(a.gguf), threshold=a.threshold, merge=not a.no_merge,
                   runtime="privacy-filter-jp.cpp libpf.so (CUDA)", postprocess="C 側: 端トリム・person 分割（不可分）; Python 側: ラベル写像・バイト→コードポイント変換" + ("・隣接結合" if not a.no_merge else ""))
        detect = lambda t: m.detect(t, merge=not a.no_merge)  # noqa: E731
        load_s = m.load_seconds
    else:
        from toramon.adapters.openai_opf import OpenAIOPF
        m = OpenAIOPF(a.checkpoint, device=a.device, decode_mode=a.decode)
        cfg.update(checkpoint=a.checkpoint, weights_sha256=sha256_file(os.path.join(a.checkpoint, "model.safetensors")),
                   decode=a.decode, runtime="opf (official PyTorch runtime)", postprocess="opf 既定: 空白トリム")
        detect = m.detect
        load_s = m.load_seconds
    cfg["load_seconds"] = load_s

    # ウォームアップ（初回起動と定常を分ける）
    t0 = time.perf_counter(); detect("ウォームアップ用の短い文です。"); cfg["warmup_ms"] = (time.perf_counter() - t0) * 1000

    lat: list[float] = []; chars = 0; failures = 0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            t0 = time.perf_counter()
            rec: dict = {"id": r["id"]}
            try:
                spans = detect(r["text"])
                rec["spans"] = [s.to_dict() for s in spans]
            except Exception as e:  # 失敗は失敗として記録し、原文を安全判定しない
                failures += 1
                rec["spans"] = []
                rec["error"] = f"{type(e).__name__}: {e}"
            ms = (time.perf_counter() - t0) * 1000
            rec["elapsed_ms"] = round(ms, 3)
            lat.append(ms); chars += len(r["text"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    lat_sorted = sorted(lat)
    cfg.update(
        finished_at=datetime.datetime.now().astimezone().isoformat(), failures=failures,
        latency_ms={"p50": statistics.median(lat) if lat else None,
                    "p95": lat_sorted[int(0.95 * (len(lat) - 1))] if lat else None,
                    "mean": statistics.fmean(lat) if lat else None},
        docs_per_sec=(len(lat) / (sum(lat) / 1000)) if lat else None,
        chars_per_sec=(chars / (sum(lat) / 1000)) if lat else None,
        total_chars=chars,
    )
    Path(a.out + ".run.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: cfg[k] for k in ("model", "n_docs", "failures", "load_seconds", "latency_ms", "docs_per_sec", "chars_per_sec")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
