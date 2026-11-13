#!/usr/bin/env python3
"""runs/speed/*.json（bench_speed.py の出力）を 1 枚の表にまとめる。判定値は 1,000 文字・同時 1 の p95。
  .venv/bin/python scripts/speed_summary.py --dir runs/speed --out runs/speed/summary.md
"""
import argparse, glob, json
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="runs/speed"); ap.add_argument("--out", default="runs/speed/summary.md"); a = ap.parse_args()
files = sorted(p for p in glob.glob(f"{a.dir}/*.json") if "prelim" not in p)
lines = ["# 速度計測（正式、同一機体・他の重い処理なし、2026-11-11）", "", "DGX Spark GB10。GGUF は libpf（CUDA）、HF/O はコンテナ内 torch（CUDA）。各セル p50 / p95 ms（n=30、ウォームアップ後）。判定値は **1,000 文字・同時 1 の p95**（受入基準 ≤ 1,000 ms）。", ""]
hdr = None; rows = []; judge = []
for f in files:
    d = json.load(open(f)); name = Path(f).stem
    runs = {(r["chars"], r["concurrency"]): r for r in d["runs"]}
    keys = sorted(runs); 
    if hdr is None:
        hdr = keys; lines += ["| 条件 | 初回ロード s | 初回呼出 ms | " + " | ".join(f"{c}字×{k}並列" for c, k in keys) + " |", "|---|---|---|" + "---|" * len(keys)]
    cells = [f"{runs[k]['p50_ms']:.0f} / {runs[k]['p95_ms']:.0f}" + (f" (失敗 {runs[k]['failures']})" if runs[k]["failures"] else "") if k in runs else "-" for k in hdr]
    lines.append(f"| {name} | {d.get('cold_load_s', 0):.1f} | {d.get('first_call_ms', 0):.0f} | " + " | ".join(cells) + " |")
    j = runs.get((1000, 1)); judge.append((name, j["p95_ms"] if j else None, runs.get((1000, 8), {}).get("docs_per_s")))
lines += ["", "## 判定値（1,000 文字・同時 1 の p95、および 1,000 文字・同時 8 のスループット）", "", "| 条件 | p95 ms | ≤1,000 ms | 同時 8 docs/s |", "|---|---|---|---|"]
for n, p95, tp in judge:
    lines.append(f"| {n} | {p95:.0f} | {'達成' if p95 is not None and p95 <= 1000 else '未達'} | {tp if tp is not None else '-'} |" if p95 is not None else f"| {n} | - | - | - |")
env = json.load(open(files[0]))["env"] if files else {}
lines += ["", "計測開始時の free -g / nvidia-smi（最初のファイル）:", "```", env.get("free_g", "").strip(), env.get("nvidia_smi", "").strip(), "```"]
Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8"); print("\n".join(lines[:12]))
