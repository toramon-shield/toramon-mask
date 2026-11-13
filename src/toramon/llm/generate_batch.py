"""Dolly ジョブを K 件ずつ 1 リクエストにまとめて、OpenAI 互換のローカル LLM サーバー（local 経路）で生成する。

接続先・モデル・API キーは環境変数 TORAMON_LLM_URL / TORAMON_LLM_MODEL / TORAMON_LLM_API_KEY で指定する（toramon.llm.clients と共通）。

generate.py（1 件 1 リクエスト）との違い:
- 共通部分（生成規則 + 保護方針）をプロンプト先頭に固定し、ジョブ固有部分（K 件）を後ろに並べる → prefill を K 件で共有。
- 出力は {"docs": [...]}。各要素は generate.py と同じスキーマ（doc_type, text, placeholders, difficulty_used）+ job_id。
  source / self_check はモデルに書かせず、こちらで source を埋める（出力トークン削減）。
- 要素ごとに検証し、合格したものだけ raw/<job_id>.json に保存（generate.py と同じレコード形式）。欠けた要素は次回再試行。
"""
from __future__ import annotations

import json, os, re, sys, time, threading, datetime, hashlib, statistics, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
POLICY = (ROOT / "docs/annotation_policy.md").read_text(encoding="utf-8")
from toramon.llm.clients import HTTP_URL as URL, ROUTES, http_api_key

KEY = http_api_key()
MODEL = ROUTES["local"][1]
NEG = {"none": "通常（個人情報を含む）", "no_pii": "負例: MASK 系を一切使わない",
       "pii_lookalike": "負例: MASK 系を使わず、KEEP 系・型番・金額・公開情報など紛らわしい情報だけ加える"}
KEEP_KINDS = {"COMPANY", "COMPANY_ADDRESS", "COMPANY_PHONE", "SHARED_EMAIL", "PUBLIC_URL", "DATE_BUSINESS", "PRODUCT_CODE", "VERSION",
              "ERROR_CODE", "AMOUNT", "INVOICE_NO", "CORP_NUMBER", "PLACE", "PUBLIC_PERSON", "REQUEST_ID"}
PH_RE = re.compile(r"\{\{([A-Z_]+?)_(\d+)(?::[A-Za-z0-9_+]+)?\}\}")


COMPACT = os.environ.get("TORAMON_BATCH_COMPACT", "1") == "1"  # placeholders 一覧を出力させず本文から抽出（出力約 -25%）


def shared_prefix() -> str:
    base = (ROOT / "prompts/dolly_pii_v1.md").read_text(encoding="utf-8")
    base = base.replace("入力は公開 Q&A データセット（llm-jp/databricks-dolly-15k-ja、CC BY-SA 3.0）の 1 レコード（instruction / context / response / category）です。",
                        "入力は公開 Q&A データセット（llm-jp/databricks-dolly-15k-ja、CC BY-SA 3.0）の複数レコード（各ジョブに instruction / context / response / category）です。")
    base = base.replace("文書を 1 件作ってください。", "文書を、ジョブごとに 1 件ずつ作ってください。ジョブ同士は無関係で、人物・番号・文脈を混ぜないこと。")
    base = base.split("## 出力 JSON 形式")[0]
    fmt = """## 出力 JSON 形式（全ジョブをまとめて 1 つの JSON）
```json
{"docs": [
  {"job_id": "dolly-000000", "doc_type": "文書種別の短い説明", "text": "本文（プレースホルダ入り。質問と回答の内容を含む）",
   "placeholders": [{"placeholder": "PERSON_1", "role": "問い合わせ顧客"}], "difficulty_used": []}
]}
```
- docs には与えられたジョブを**全件、与えられた順に**入れる。各要素の job_id はジョブの job_id と一致させる。
- 各文書は単独で読める完結した文書にする（他ジョブへの言及禁止）。文書長はジョブごとの目安に従い、件数が多くても短くしない。
"""
    if COMPACT:
        fmt = fmt.replace(""",
   "placeholders": [{"placeholder": "PERSON_1", "role": "問い合わせ顧客"}], "difficulty_used": []}""", """, "difficulty_used": []}""")
        fmt += "- placeholders 一覧は出力しない（本文中のプレースホルダから自動で集計する）。\n"
    return base + fmt + "\n## 保護方針（参考・抜粋）\n" + POLICY[:4000] + "\n"


def job_block(job: dict) -> str:
    d = job["dolly"]
    kinds = ", ".join(f"{{{{{k}_n}}}}" for k in job["mask_kinds"]) or "（なし: 負例）"
    keeps = ", ".join(f"{{{{{k}_n}}}}" for k in job["keep_kinds"]) or "（任意）"
    return f"""
### ジョブ job_id: {job['job_id']}
- category: {d['category']} / 登場する架空の個人の人数: {job['n_people']}
- 使う MASK 系プレースホルダ（番号は人物・実体ごと。全種類を必ず使う）: {kinds}
- 使う KEEP 系プレースホルダ: {keeps}
- 難易度軸: {', '.join(job['difficulty_axes']) or '（指定なし）'} / 負例条件: {NEG[job['negative_type']]} / 負例軸: {', '.join(job['negative_axes']) or '（指定なし）'}
- 文書長の目安: {job['length_hint']}
#### instruction
{d['instruction']}
#### context
{d['context'] or '（なし）'}
#### response
{d['response']}
"""


def build_prompt(jobs: list[dict]) -> str:
    return (shared_prefix() + f"\n## 今回のジョブ（{len(jobs)} 件）\n" + "".join(job_block(j) for j in jobs)
            + f"\nJSON のみを出力してください。docs は {len(jobs)} 要素。")


def extract_json(text: str):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    cands = [m.group(1)] if m else []
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j > i: cands.append(text[i:j + 1])
    for c in cands:
        for s in (c, re.sub(r",\s*([}\]])", r"\1", c)):
            try: return json.loads(s)
            except json.JSONDecodeError: pass
    return None


def quality(job: dict, o: dict) -> dict:
    """自動品質指標（要素単位）。"""
    t = o.get("text") or ""
    used = {m.group(1) for m in PH_RE.finditer(t)}
    mask_used = {k for k in used if k not in KEEP_KINDS}
    if job["negative_type"] == "none":
        mask_ok = set(job["mask_kinds"]) <= used
    else:
        mask_ok = not mask_used
    keep_ok = set(job["keep_kinds"]) <= used
    resp = re.sub(r"\s", "", job["dolly"]["response"]); tt = re.sub(r"\s", "", t)
    grams = {resp[i:i + 4] for i in range(max(0, len(resp) - 3))}
    recall = (sum(g in tt for g in grams) / len(grams)) if grams else 1.0
    return {"mask_ok": mask_ok, "keep_ok": keep_ok, "len": len(t), "len_ok": 400 <= len(t) <= 1500, "resp_recall4": round(recall, 3)}


def call(prompt: str, max_tokens: int, timeout_s: int):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.8, "top_p": 0.95,
                       "max_tokens": max_tokens, "stream": False,
                       "chat_template_kwargs": {"thinking": False, "enable_thinking": False}}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {KEY}"} if KEY else {})})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"].get("content") or "", data.get("usage") or {}, data["choices"][0].get("finish_reason")


def run(jobs_path: str, out_dir: str, batch: int, workers: int, limit: int, timeout_s: int, tok_per_doc: int, stats_path: str | None):
    jobs = [json.loads(l) for l in open(jobs_path, encoding="utf-8") if l.strip()]
    jobs = [j for j in jobs if j["generator_route"] == "local"]
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); logd = out / "llm_logs_batch"; logd.mkdir(exist_ok=True)
    todo = [j for j in jobs if not (out / f"{j['job_id']}.json").exists()]
    if limit: todo = todo[:limit]
    groups = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    lock = threading.Lock(); agg = {"ok": 0, "missing": 0, "req": 0, "req_fail": 0, "prompt_tok": 0, "gen_tok": 0, "q": []}
    t_start = time.time()
    print(f"todo={len(todo)} batch={batch} workers={workers} requests={len(groups)}", file=sys.stderr, flush=True)

    def work(g: list[dict]):
        prompt = build_prompt(g); t0 = time.time(); started = datetime.datetime.now().astimezone().isoformat()
        try:
            text, usage, fin = call(prompt, min(max(tok_per_doc, 1800) * len(g) + 200, 7900), timeout_s); err = None
        except Exception as e:  # noqa: BLE001
            text, usage, fin, err = "", {}, None, repr(e)[:300]
        el = round(time.time() - t0, 2)
        parsed = extract_json(text) if text else None
        docs = parsed.get("docs") if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else None)
        byid = {}
        for d in docs or []:
            if isinstance(d, dict) and isinstance(d.get("text"), str) and d["text"].strip() and (COMPACT or "placeholders" in d):
                byid.setdefault(str(d.get("job_id", "")), d)
        tag = f"{g[0]['job_id']}_x{len(g)}"
        (logd / f"{tag}.raw.txt").write_text(text, encoding="utf-8")
        okn = 0; qs = []
        for j in g:
            d = byid.get(j["job_id"])
            if d is None: continue
            d = {k: v for k, v in d.items() if k != "job_id"}
            if COMPACT or not isinstance(d.get("placeholders"), list):
                seen = dict.fromkeys(f"{m.group(1)}_{m.group(2)}" for m in PH_RE.finditer(d["text"]))
                d["placeholders"] = [{"placeholder": k, "role": ""} for k in seen]
            d.setdefault("difficulty_used", []); d["source"] = {"dolly_index": j["dolly_index"], "category": j["dolly"]["category"]}
            q = quality(j, d); qs.append(q)
            meta = {"route": "local", "tool": "http-batch", "model": MODEL, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "started_at": started, "elapsed_s": el, "exit_code": 0, "output_chars": len(text), "attempts": 1, "json_ok": True,
                    "raw_path": str(logd / f"{tag}.raw.txt"), "batch_size": len(g), "compact": COMPACT, "finish_reason": fin, "usage": usage, "quality": q, "error": None}
            (out / f"{j['job_id']}.json").write_text(json.dumps({"job": j, "llm": meta, "output": d}, ensure_ascii=False, indent=1), encoding="utf-8")
            okn += 1
        with lock:
            agg["req"] += 1; agg["req_fail"] += okn == 0; agg["ok"] += okn; agg["missing"] += len(g) - okn
            agg["prompt_tok"] += usage.get("prompt_tokens", 0); agg["gen_tok"] += usage.get("completion_tokens", 0); agg["q"] += qs
            rate = agg["ok"] / max(1e-9, time.time() - t_start) * 3600
        print(f"{tag}: ok={okn}/{len(g)} {el}s fin={fin} ptok={usage.get('prompt_tokens')} gtok={usage.get('completion_tokens')} "
              f"err={err} | total ok={agg['ok']} rate={rate:.0f}/h", file=sys.stderr, flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(work, g) for g in groups]): f.result()
    wall = time.time() - t_start; q = agg.pop("q")
    summ = dict(agg, batch=batch, workers=workers, wall_s=round(wall, 1), docs_per_h=round(agg["ok"] / wall * 3600, 1),
                mask_ok=round(statistics.mean(x["mask_ok"] for x in q), 3) if q else None,
                keep_ok=round(statistics.mean(x["keep_ok"] for x in q), 3) if q else None,
                len_ok=round(statistics.mean(x["len_ok"] for x in q), 3) if q else None,
                len_med=statistics.median(x["len"] for x in q) if q else None,
                resp_recall4=round(statistics.mean(x["resp_recall4"] for x in q), 3) if q else None)
    print(json.dumps(summ, ensure_ascii=False))
    if stats_path:
        with open(stats_path, "a", encoding="utf-8") as f: f.write(json.dumps(summ, ensure_ascii=False) + "\n")
    return summ


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="data/dolly/jobs.jsonl"); ap.add_argument("--out", default="data/dolly/raw")
    ap.add_argument("--batch", type=int, default=4); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--tok-per-doc", type=int, default=1100); ap.add_argument("--stats", default=None)
    a = ap.parse_args()
    run(a.jobs, a.out, a.batch, a.workers, a.limit, a.timeout, a.tok_per_doc, a.stats)
