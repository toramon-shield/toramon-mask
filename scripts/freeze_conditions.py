#!/usr/bin/env python3
"""最終条件の固定。候補モデル・閾値・デコード設定・ルール版・注釈方針・評価コード・受入基準のハッシュを frozen_conditions.json に記録する。

  .venv/bin/python scripts/freeze_conditions.py --candidate T-O=runs/.../model --candidate T-S=runs/.../model --baseline S=<f16.gguf> --baseline O=<opf dir> \
      --threshold S=0.5 --threshold T-S=0.5 --reason "dev 選定基準 (configs/acceptance.yaml selection) に基づく" --out configs/frozen_conditions.json
凍結後に acceptance.yaml を変更する場合は理由と日付を残す（目標は下げない）。
"""
import argparse, hashlib, json, datetime, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    if p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.is_file(): h.update(f.name.encode()); h.update(sha(f).encode())
        return h.hexdigest()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""): h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--candidate", action="append", default=[]); ap.add_argument("--baseline", action="append", default=[])
    ap.add_argument("--threshold", action="append", default=[]); ap.add_argument("--reason", required=True); ap.add_argument("--out", default=str(ROOT / "configs/frozen_conditions.json"))
    a = ap.parse_args()
    out = {"frozen_at": datetime.datetime.now().astimezone().isoformat(), "reason": a.reason, "models": {}, "thresholds": dict(kv.split("=", 1) for kv in a.threshold),
           "artifacts": {}}
    for kv in a.candidate + a.baseline:
        k, v = kv.split("=", 1); p = Path(v)
        out["models"][k] = {"path": str(p), "sha256": sha(p) if p.exists() else None, "role": "candidate" if kv in a.candidate else "baseline"}
    # 受入基準の状態を frozen にしてからハッシュを取る（順序を逆にすると final_eval の改変検査で自分自身が引っかかる）
    acc = ROOT / "configs/acceptance.yaml"; txt = acc.read_text(encoding="utf-8")
    if "status: draft" in txt:
        txt = txt.replace("status: draft", "status: frozen").replace("frozen_at: null", f"frozen_at: \"{out['frozen_at']}\"")
        acc.write_text(txt, encoding="utf-8"); out["acceptance_frozen"] = True
    for rel in ["configs/rules_R.yaml", "configs/label_mapping.yaml", "configs/acceptance.yaml", "docs/annotation_policy.md", "src/toramon/evaluate.py", "src/toramon/spans.py",
                "src/toramon/rules/engine.py", "src/toramon/filter.py", "src/toramon/adapters/sumeshi_pf.py", "src/toramon/adapters/openai_opf.py", "src/toramon/adapters/hf_token_cls.py"]:
        out["artifacts"][rel] = sha(ROOT / rel)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "artifacts"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
