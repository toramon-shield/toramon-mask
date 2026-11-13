#!/usr/bin/env python3
"""ローカルフィルタ CLI。標準入力のテキストを検出・置換して出力する（外部通信なし）。

  echo "..." | python scripts/toramon_cli.py redact --backend S --gguf <path>
  echo "..." | python scripts/toramon_cli.py detect --backend S --gguf <path> --json
  python scripts/toramon_cli.py serve --backend S --gguf <path> --port 8765   # ローカル API（FastAPI）
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toramon.filter import ToramonFilter, Policy  # noqa: E402


def build(a) -> ToramonFilter:
    if a.backend == "S":
        from toramon.adapters.sumeshi_pf import SumeshiPF
        det = SumeshiPF(a.gguf, device=a.device, threshold=a.threshold)
        mv = f"{a.backend}:{Path(a.gguf).name}"
    elif a.gguf:  # T-O / T-S の GGUF: 生ロジットを Toramon 復号器（HF adapter と同一）で復号する。libpf 既定の復号器では T-O のラベルが写像されない
        from toramon.adapters.sumeshi_pf import SumeshiPFDecoded
        det = SumeshiPFDecoded(a.gguf, device=a.device, threshold=a.threshold, label_space="identity" if a.backend == "T-O" else "S")
        mv = f"{a.backend}:{Path(a.gguf).name}:toramon-decode"
    else:
        from toramon.adapters.openai_opf import OpenAIOPF
        det = OpenAIOPF(a.checkpoint, device=a.device)
        mv = f"{a.backend}:{Path(a.checkpoint).name}"
    pol = Policy(use_rules=not a.no_rules, use_boundary_fix=not a.no_rules, use_keep_suppression=not a.no_rules, placeholder_style=a.style)
    return ToramonFilter(det, mv, policy=pol)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["detect", "redact", "serve"])
    ap.add_argument("--backend", default="S", choices=["S", "O", "T-O", "T-S"])
    ap.add_argument("--gguf"); ap.add_argument("--checkpoint"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--threshold", type=float, default=0.5); ap.add_argument("--no-rules", action="store_true")
    ap.add_argument("--style", default="label", choices=["label", "asterisk"]); ap.add_argument("--json", action="store_true")
    ap.add_argument("--port", type=int, default=8765); ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    f = build(a)
    if a.cmd == "serve":
        from toramon.api import serve
        serve(f, a.host, a.port); return 0
    text = sys.stdin.read()
    res = f.redact(text) if a.cmd == "redact" else f.detect(text)
    if a.json or a.cmd == "detect":
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=1))
    else:
        if res.status != "ok":
            print(json.dumps(res.to_dict(), ensure_ascii=False), file=sys.stderr); return 1
        sys.stdout.write(res.redacted_text)
    return 0 if res.status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
