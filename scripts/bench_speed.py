#!/usr/bin/env python3
"""推論速度の計測。同一機体・同一入力で、文字数帯（100/1,000/5,000/10,000）× 同時数（1/4/8）、ウォームアップと初回起動を分ける。

  python scripts/bench_speed.py --backend S --gguf <path> --out runs/speed/S.json
  env/run.sh python scripts/bench_speed.py --backend O --checkpoint /models/openai-privacy-filter/original --out runs/speed/O.json
  env/run.sh python scripts/bench_speed.py --backend HF --checkpoint <dir> --label-space S --out ...
入力は合成文書（data/speed_inputs.jsonl があればそれを、なければ合成テンプレートを繰り返して生成）。失敗率も記録。
注意: 計測中は同じ機体で学習や他の重い処理を動かさない。free -g と nvidia-smi の状態を同時に記録する。
"""
from __future__ import annotations
import argparse, json, statistics, subprocess, sys, time, datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

BASE = ("お世話になっております。{c}株式会社の{n}です。先日の件につきまして、ご担当の佐々木様（電話 090-{a:04d}-{b:04d}、メール sasaki{a}@example.jp）へ"
        "ご連絡いたしました。納品先は〒{a:03d}-{b:04d} 東京都千代田区丸の内{z}丁目{a}番{b}号 サンプルビル{z}F、会員番号 M-{a:04d}{b:02d} です。請求額は12,000円、"
        "型番 AB-{b:04d}、バージョン 2.{z}.1 を予定しています。\n")


def make_text(n_chars: int, seed: int = 0) -> str:
    out = []; i = seed
    while sum(len(x) for x in out) < n_chars:
        out.append(BASE.format(c=["みらい", "さくら", "ひかり"][i % 3], n=["山田", "田中", "鈴木"][i % 3], a=(1000 + i * 37) % 10000, b=(2000 + i * 91) % 10000, z=1 + i % 9)); i += 1
    return "".join(out)[:n_chars]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["S", "O", "HF"]); ap.add_argument("--gguf"); ap.add_argument("--checkpoint"); ap.add_argument("--label-space", default="identity"); ap.add_argument("--decode", default="libpf", choices=["libpf", "toramon"])
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", required=True); ap.add_argument("--sizes", default="100,1000,5000,10000")
    ap.add_argument("--concurrency", default="1,4,8"); ap.add_argument("--n", type=int, default=30); ap.add_argument("--with-rules", action="store_true")
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]; concs = [int(x) for x in a.concurrency.split(",")]
    env = {"started_at": datetime.datetime.now().astimezone().isoformat()}
    try:
        env["free_g"] = subprocess.run(["free", "-g"], capture_output=True, text=True).stdout
        env["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True).stdout
    except Exception:
        pass
    t0 = time.perf_counter()
    if a.backend == "S" and a.decode == "toramon":
        from toramon.adapters.sumeshi_pf import SumeshiPFDecoded
        det = SumeshiPFDecoded(a.gguf, device=a.device, label_space=a.label_space); detect = det.detect
    elif a.backend == "S":
        from toramon.adapters.sumeshi_pf import SumeshiPF
        det = SumeshiPF(a.gguf, device=a.device); detect = det.detect
    elif a.backend == "O":
        from toramon.adapters.openai_opf import OpenAIOPF
        det = OpenAIOPF(a.checkpoint, device=a.device); detect = det.detect
    else:
        from toramon.adapters.hf_token_cls import HFTokenClassifier
        det = HFTokenClassifier(a.checkpoint, device=a.device, label_space=a.label_space); detect = det.detect
    cold_load_s = time.perf_counter() - t0
    if a.with_rules:
        from toramon.filter import ToramonFilter
        f = ToramonFilter(det, a.backend); detect = lambda t: f.detect(t).spans  # noqa: E731
    # 初回呼び出し（コールド）
    t0 = time.perf_counter(); detect(make_text(1000)); first_call_ms = (time.perf_counter() - t0) * 1000
    for _ in range(3): detect(make_text(1000))
    results = {"backend": a.backend, "device": a.device, "cold_load_s": round(cold_load_s, 2), "first_call_ms": round(first_call_ms, 1), "env": env, "with_rules": a.with_rules, "runs": []}
    for size in sizes:
        texts = [make_text(size, seed=i) for i in range(a.n)]
        for c in concs:
            lat = []; fails = 0
            def one(t):
                s = time.perf_counter()
                try: detect(t); return (time.perf_counter() - s) * 1000, False
                except Exception: return (time.perf_counter() - s) * 1000, True
            wall0 = time.perf_counter()
            if c == 1:
                for t in texts:
                    ms, fl = one(t); lat.append(ms); fails += fl
            else:
                with ThreadPoolExecutor(max_workers=c) as ex:
                    for ms, fl in ex.map(one, texts): lat.append(ms); fails += fl
            wall = time.perf_counter() - wall0
            ls = sorted(lat)
            results["runs"].append({"chars": size, "concurrency": c, "n": a.n, "p50_ms": round(statistics.median(lat), 1), "p95_ms": round(ls[int(0.95 * (len(ls) - 1))], 1),
                                    "mean_ms": round(statistics.fmean(lat), 1), "docs_per_s": round(a.n / wall, 2), "chars_per_s": round(a.n * size / wall, 0), "failures": fails})
            print(results["runs"][-1], flush=True)
    try:
        import torch
        results["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None
    except Exception:
        pass
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
