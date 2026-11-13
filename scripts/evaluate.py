#!/usr/bin/env python3
"""予測 JSONL を正解 JSONL に対して採点する。

  python scripts/evaluate.py --gold gold.jsonl --pred pred.jsonl [--out result.json]
  python scripts/evaluate.py --gold gold.jsonl --pred A.jsonl --compare B.jsonl   # 対応付き差分 CI（A - B）
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toramon.evaluate import evaluate_files, compare_files, format_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True); ap.add_argument("--pred", required=True)
    ap.add_argument("--compare"); ap.add_argument("--out"); ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    if a.compare:
        res = compare_files(a.gold, a.pred, a.compare, a.n_boot)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        res = evaluate_files(a.gold, a.pred)
        print(format_report(res))
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
