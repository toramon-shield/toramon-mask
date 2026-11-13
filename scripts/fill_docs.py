#!/usr/bin/env python3
"""LLM のプレースホルダ文書（data/<split>/raw/*.json）に合成値を挿入し、Toramon JSONL（正解付き）を出力する。

  .venv/bin/python scripts/fill_docs.py --raw data/pilot/raw --split pilot --pool train --out data/pilot/filled.jsonl --seed 7

- 位置はプログラムが確定（コードポイント）。schema 検証と validate_spans を全件で行い、失敗は data/<split>/quarantine.jsonl に理由付きで隔離。
- LLM が本文に生の PII らしき値（メール形式、電話形式、〒住所）を直接書いていないかも検査し、該当文書は隔離（プレースホルダ以外の保護対象漏れの機械検査。独立レビューでも再検査する）。
"""
from __future__ import annotations
import argparse, json, os, random, re, sys, datetime
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import jsonschema  # noqa: E402
from toramon.datagen.pools import Pool  # noqa: E402
from toramon.datagen.fill import fill_document, FillError  # noqa: E402
from toramon.spans import Span, validate_spans  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.load(open(ROOT / "configs/data_schema.json", encoding="utf-8"))
ROUTE_MODEL = {"gpt": "gpt-6.1-terra", "gemini": "gemini-3.8-flash-high", "claude": "claude-sonnet-4-6", "claude_sub": "claude (coding-agent subagent)", "local": os.environ.get("TORAMON_LLM_MODEL", "local-model") + " (OpenAI-compatible local server)"}
LITERAL_PII = [re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+\.[A-Za-z]{2,}"), re.compile(r"(?<!\d)0\d{1,3}[\-－]\d{2,4}[\-－]\d{3,4}(?!\d)"), re.compile(r"〒\s?\d{3}[\-－]?\d{4}")]


def length_band(n: int) -> str:
    return "short" if n < 200 else "normal" if n < 2000 else "long" if n < 10000 else "xlong"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--raw", required=True); ap.add_argument("--split", required=True); ap.add_argument("--pool", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=7); ap.add_argument("--pipeline-version", default="fill-0.1")
    ap.add_argument("--keep-existing", action="store_true", help="既存 --out の行はそのまま保持し、未収録の raw だけを追加で穴埋めする（レビュー済み本文を変えないため）")
    a = ap.parse_args()
    pool = Pool(a.pool, seed=a.seed)
    files = sorted(Path(a.raw).glob("*.json"))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    qpath = out.parent / "quarantine.jsonl"
    existing = []; existing_ids = set()
    if a.keep_existing and out.exists():
        existing = [l for l in open(out, encoding="utf-8") if l.strip()]
        existing_ids = {json.loads(l)["id"] for l in existing}
        if qpath.exists():
            existing_ids |= {json.loads(l)["job_id"].lower() for l in open(qpath, encoding="utf-8") if l.strip()}
    stats = Counter(); per_route = Counter(); ents_by_label = Counter(); ents_by_action = Counter()
    stats["kept_existing"] = len(existing)
    with open(out, "w", encoding="utf-8") as fo, open(qpath, "a" if a.keep_existing else "w", encoding="utf-8") as fq:
        for l in existing: fo.write(l if l.endswith("\n") else l + "\n")
        for i, p in enumerate(files):
            rec = json.loads(p.read_text(encoding="utf-8")); job = rec["job"]; o = rec.get("output")
            if job["job_id"].lower() in existing_ids: continue
            route = job["generator_route"]
            def quarantine(reason, extra=None):
                stats["quarantined"] += 1; per_route[f"{route}:q"] += 1
                fq.write(json.dumps({"job_id": job["job_id"], "route": route, "reason": reason, "extra": extra}, ensure_ascii=False) + "\n")
            if not o or not isinstance(o, dict) or not isinstance(o.get("text"), str) or not o["text"].strip():
                quarantine("no_json_or_text"); continue
            ph = {}
            for item in o.get("placeholders") or []:
                if isinstance(item, dict) and item.get("placeholder"):
                    ph[item["placeholder"]] = {k: v for k, v in item.items() if k != "placeholder"}
            rng = random.Random(f"{a.seed}:{job['job_id']}")
            tmpl = o["text"]; fill_notes = []
            fd = None
            for attempt in range(6):
                try:
                    fd = fill_document(tmpl, ph, pool, random.Random(f"{a.seed}:{job['job_id']}")); break
                except FillError as e:
                    m = re.search(r"\{\{([A-Z_]+_?\d*):([A-Za-z0-9_+]+)\}\}: unknown variant", str(e))
                    if m:  # LLM が仕様外の variant を使った → 既定表記へ落として再試行（記録）
                        tmpl = tmpl.replace("{{" + m.group(1) + ":" + m.group(2) + "}}", "{{" + m.group(1) + "}}"); fill_notes.append(f"dropped_variant:{m.group(1)}:{m.group(2)}"); continue
                    quarantine("fill_error", str(e)[:300]); break
            if fd is None:
                if not fill_notes or attempt == 5: 
                    if fill_notes: quarantine("fill_error_after_variant_fallback", fill_notes)
                continue
            text = fd.text; ents = fd.entities
            if job["negative_type"] != "none" and any(e["action"] == "MASK" for e in ents):
                quarantine("negative_doc_contains_mask_placeholder", [e["text"] for e in ents if e["action"] == "MASK"][:5]); continue
            # 生 PII らしき直書きの検査（プレースホルダ由来の span 以外の場所）
            covered = [False] * len(text)
            for e in ents:
                for k in range(e["start"], e["end"]): covered[k] = True
            leaks = []
            for rx in LITERAL_PII:
                for m in rx.finditer(text):
                    if not any(covered[k] for k in range(m.start(), m.end())):
                        leaks.append(m.group(0))
            if leaks:
                quarantine("literal_pii_like_outside_placeholders", leaks[:5]); continue
            probs = validate_spans(text, [Span(e["start"], e["end"], e["label"], e["text"], e["action"], e.get("entity_id", "")) for e in ents])
            if probs:
                quarantine("span_validation", probs[:3]); continue
            doc = {"id": job['job_id'].lower(), "split": a.split, "scenario_family": job["scenario_family"],
                   "template_family": f"{job['scenario_family']}|{o.get('doc_type', '')[:40]}", "value_pool": f"pool_{a.pool}",
                   "generator_model": ROUTE_MODEL[route], "generator_route": route, "prompt_version": job["prompt_version"], "domain": job["domain"],
                   "doc_type": str(o.get("doc_type", ""))[:120], "length_band": length_band(len(text)), "difficulty": "hard" if len(job["difficulty_axes"]) >= 2 else ("medium" if job["difficulty_axes"] else "easy"),
                   "negative_type": job["negative_type"], "text": text, "entities": ents, "review_status": "unreviewed",
                   "provenance": {"created_at": datetime.datetime.now().astimezone().isoformat(), "pipeline_version": a.pipeline_version, "synthetic": True,
                                  "generation_job_id": job["job_id"], "temperature": None, "seed": None, "tokens_in": None, "tokens_out": rec["llm"].get("output_chars"),
                                  "cost_jpy": None, "retries": max(0, rec["llm"].get("attempts", 1) - 1), "placeholder_doc_ref": str(p),
                                  "notes": f"fill_notes={fill_notes} difficulty_axes={job['difficulty_axes']} negative_axes={job['negative_axes']} keep_values={len(fd.keep_values)} llm_elapsed_s={rec['llm'].get('elapsed_s')}"}}
            try:
                jsonschema.validate(doc, SCHEMA)
            except jsonschema.ValidationError as e:
                quarantine("schema", e.message[:300]); continue
            fo.write(json.dumps(doc, ensure_ascii=False) + "\n")
            stats["ok"] += 1; per_route[route] += 1
            for e in ents: ents_by_label[e["label"]] += 1; ents_by_action[e["action"]] += 1
            stats[f"len:{doc['length_band']}"] += 1; stats[f"neg:{doc['negative_type']}"] += 1
    print(json.dumps({"files": len(files), "stats": dict(stats), "per_route": dict(per_route), "entities_by_label": dict(ents_by_label), "entities_by_action": dict(ents_by_action), "out": str(out), "quarantine": str(qpath)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
