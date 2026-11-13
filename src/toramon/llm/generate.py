"""ジョブ仕様 → 生成プロンプト → LLM 呼び出し → プレースホルダ文書の保存。

出力: data/<split>/raw/<job_id>.json （LLM の JSON 出力 + ジョブ仕様 + 呼び出しメタ）
値の挿入（fill）は別工程（datagen.fill）。
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .clients import call_llm

ROOT = Path(__file__).resolve().parents[3]
GEN_PROMPT = (ROOT / "prompts/gen_v1.md").read_text(encoding="utf-8")
POLICY = (ROOT / "docs/annotation_policy.md").read_text(encoding="utf-8")


DOLLY_PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts/dolly_pii_v1.md"


def build_dolly_prompt(job: dict) -> str:
    d = job["dolly"]
    kinds = ", ".join(f"{{{{{k}_n}}}}" for k in job["mask_kinds"]) or "（なし: 負例）"
    keeps = ", ".join(f"{{{{{k}_n}}}}" for k in job["keep_kinds"]) or "（任意）"
    neg = {"none": "通常（個人情報を含む）", "no_pii": "負例: MASK 系を一切使わない", "pii_lookalike": "負例: MASK 系を使わず、KEEP 系・型番・金額・公開情報など紛らわしい情報だけ加える"}[job["negative_type"]]
    spec = f"""
## 今回のジョブ
- job_id: {job['job_id']} / dolly_index: {job['dolly_index']} / category: {d['category']}
- 登場する架空の個人の人数: {job['n_people']}
- 使う MASK 系プレースホルダ（番号は人物・実体ごと。全種類を必ず使う）: {kinds}
- 使う KEEP 系プレースホルダ: {keeps}
- 難易度軸: {', '.join(job['difficulty_axes']) or '（指定なし）'}
- 負例条件: {neg}
- 負例軸: {', '.join(job['negative_axes']) or '（指定なし）'}
- 文書長の目安: {job['length_hint']}

## 元レコード（llm-jp/databricks-dolly-15k-ja）
### instruction
{d['instruction']}
### context
{d['context'] or '（なし）'}
### response
{d['response']}

## 保護方針（参考・抜粋）
{POLICY[:4000]}
"""
    return DOLLY_PROMPT_PATH.read_text(encoding="utf-8") + "\n" + spec + "\nJSON のみを出力してください。source には {\"dolly_index\": " + str(job["dolly_index"]) + ", \"category\": \"" + d["category"] + "\"} を入れてください。"


def build_prompt(job: dict) -> str:
    if job.get("prompt_version") == "dolly_pii_v1":
        return build_dolly_prompt(job)
    kinds = ", ".join(f"{{{{{k}_n}}}}" for k in job["mask_kinds"]) or "（なし）"
    keeps = ", ".join(f"{{{{{k}_n}}}}" for k in job["keep_kinds"]) or "（任意）"
    neg = {"none": "通常（個人情報を含む）", "no_pii": "負例: 個人情報を一切含めない", "pii_lookalike": "負例: 個人情報に似た表現（KEEP 系・型番・金額・公開情報）のみを含め、MASK 系は含めない"}[job["negative_type"]]
    spec = f"""
## 今回の生成仕様
- 担当 split: {job['split']}（開発用パイロット）
- シナリオ群: {job['scenario_family']} — {job['family_desc']}
- 業務分野: {job['domain']}
- 文書長: {job['length_hint']}
- 登場する個人の人数: {job['n_people']}
- 使う MASK 系プレースホルダ（番号 n は人物・実体ごとに振る。全種類を必ず使い、必要なら複数回・複数人分使ってよい）: {kinds}
- 使う KEEP 系プレースホルダ（紛らわしい残すべき情報）: {keeps}
- 難易度軸: {', '.join(job['difficulty_axes']) or '（指定なし）'}
- 負例条件: {neg}
- 負例軸: {', '.join(job['negative_axes']) or '（指定なし）'}

## 保護方針（参考・抜粋）
{POLICY[:6000]}
"""
    return GEN_PROMPT + "\n" + spec + "\nJSON のみを出力してください。"


def run_jobs(jobs_path: str, out_dir: str, routes: tuple[str, ...] = ("gpt", "gemini", "claude"), limit: int = 0,
             workers: int = 3, timeout_s: int = 600) -> dict:
    jobs = [json.loads(l) for l in open(jobs_path, encoding="utf-8") if l.strip()]
    jobs = [j for j in jobs if j["generator_route"] in routes]
    if limit:
        jobs = jobs[:limit]
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    log_dir = out / "llm_logs"
    todo = [j for j in jobs if not (out / f"{j['job_id']}.json").exists()]
    stats = {"total": len(jobs), "skipped_existing": len(jobs) - len(todo), "ok": 0, "json_fail": 0, "elapsed_s": {}}

    def work(job: dict):
        prompt = build_prompt(job)
        parsed, meta = call_llm(job["generator_route"], prompt, log_dir, job["job_id"], timeout_s=timeout_s, required_keys=("text", "placeholders"))
        rec = {"job": job, "llm": meta.__dict__, "output": parsed}
        # 失敗（JSON なし・エラー払い出し）は raw_failed/ に置き、raw/ には書かない（次回実行で再試行できるようにする）
        dest = out if parsed is not None else (out.parent / (out.name + "_failed"))
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"{job['job_id']}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        return job["job_id"], meta

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, j) for j in todo]
        for fut in as_completed(futs):
            jid, meta = fut.result()
            stats["ok" if meta.json_ok else "json_fail"] += 1
            stats["elapsed_s"].setdefault(meta.route, []).append(meta.elapsed_s)
            print(f"{jid}: json_ok={meta.json_ok} {meta.elapsed_s}s attempts={meta.attempts} rc={meta.exit_code}", file=sys.stderr, flush=True)
    return stats


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--routes", default="gpt,gemini,claude"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3); ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()
    st = run_jobs(a.jobs, a.out, tuple(a.routes.split(",")), a.limit, a.workers, a.timeout)
    import statistics
    for r, v in st["elapsed_s"].items():
        st["elapsed_s"][r] = {"n": len(v), "median": statistics.median(v), "max": max(v)}
    print(json.dumps(st, ensure_ascii=False))
