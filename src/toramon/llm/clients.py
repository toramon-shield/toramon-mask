"""生成・レビュー担当 LLM の非対話クライアント。

経路（route）は 4 種類:

- ``gpt``    : OpenAI Codex CLI（``codex exec``）
- ``gemini`` : Antigravity CLI（``agy --print``）
- ``claude`` : Antigravity CLI 経由の Claude
- ``local``  : OpenAI 互換の chat/completions サーバー（vLLM / llama.cpp server / Ollama など、自前で用意したもの）

呼び出しごとに prompt_sha256、モデル ID、経路、所要時間、出力文字数、再試行、終了コードを JSON ログに残す。
温度・seed は CLI から制御できないため null で記録する。出力は JSON を期待し、コードフェンスや前後の説明を剥がして抽出する。

環境変数:

- TORAMON_MODEL_GPT / TORAMON_MODEL_GEMINI / TORAMON_MODEL_CLAUDE: 各 CLI 経路のモデル ID（既定値は下の ROUTES）
- TORAMON_LLM_URL:      ``local`` 経路の URL（例: http://127.0.0.1:8000/v1/chat/completions）
- TORAMON_LLM_MODEL:    ``local`` 経路のモデル名
- TORAMON_LLM_API_KEY:  ``local`` 経路の API キー（不要なら未設定）。ファイルから読む場合は TORAMON_LLM_API_KEY_FILE
- TORAMON_LLM_MAX_TOKENS / TORAMON_LLM_THINKING: ``local`` 経路の出力上限と思考モード（"1" で有効）
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path

ROUTES = {
    # 論理名 -> (ツール, モデルID)
    "gpt": ("codex", os.environ.get("TORAMON_MODEL_GPT", "gpt-6.1-terra")),
    "gemini": ("agy", os.environ.get("TORAMON_MODEL_GEMINI", "gemini-3.8-flash-high")),
    "claude": ("agy", os.environ.get("TORAMON_MODEL_CLAUDE", "claude-sonnet-4-6")),
    "local": ("http", os.environ.get("TORAMON_LLM_MODEL", "local-model")),
}


@dataclass
class LLMResult:
    route: str
    tool: str
    model: str
    prompt_sha256: str
    started_at: str
    elapsed_s: float
    exit_code: int
    output_chars: int
    attempts: int
    json_ok: bool
    raw_path: str
    temperature: float | None = None
    seed: int | None = None
    error: str | None = None


def _extract_json(text: str) -> dict | list | None:
    """モデル出力から最初の JSON オブジェクトを取り出す（フェンス・前置き・後置きを許容）。"""
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.S)
    cands = [m.group(1)] if m else []
    # フェンスなし: 最初の { から最後の } まで
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j > i:
        cands.append(text[i : j + 1])
    for c in cands:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            # 末尾カンマ等の軽微な崩れを修復
            c2 = re.sub(r",\s*([}\]])", r"\1", c)
            try:
                return json.loads(c2)
            except json.JSONDecodeError:
                continue
    return None


ERROR_KEYS = {"short_error", "error_code", "AGY_ERROR", "status"}


def is_error_payload(obj) -> str | None:
    """CLI がエラー JSON（例: agy の RESOURCE_EXHAUSTED 429）を返した場合にその要約を返す。"""
    if isinstance(obj, dict) and ("short_error" in obj or "error_code" in obj) and "text" not in obj and "findings" not in obj:
        return f"{obj.get('status') or ''} {obj.get('error_code') or ''}: {str(obj.get('short_error') or '')[:200]}".strip()
    return None


HTTP_URL = os.environ.get("TORAMON_LLM_URL", "http://127.0.0.1:8000/v1/chat/completions")
HTTP_MAX_TOKENS = int(os.environ.get("TORAMON_LLM_MAX_TOKENS", "6000"))
HTTP_THINKING = os.environ.get("TORAMON_LLM_THINKING", "0") == "1"  # 既定で思考を無効化（速度優先）


def http_api_key() -> str:
    """local 経路の API キー。環境変数 → キーファイルの順に探し、どちらもなければ空文字（認証なしサーバー向け）。"""
    if os.environ.get("TORAMON_LLM_API_KEY"):
        return os.environ["TORAMON_LLM_API_KEY"].strip()
    kf = os.environ.get("TORAMON_LLM_API_KEY_FILE")
    if kf and Path(kf).exists():
        return Path(kf).read_text(encoding="utf-8").strip()
    return ""


def _run_cli(tool: str, model: str, prompt: str, timeout_s: int, wd: str) -> tuple[int, str, str | None]:
    """codex / agy を読み取り専用・非対話で呼び、最終メッセージを返す。"""
    if tool == "codex":
        fd, last = tempfile.mkstemp(prefix="toramon-codex-"); os.close(fd)
        cmd = ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "--color", "never", "-C", wd,
               "-s", "read-only", "-m", model, "-o", last, prompt]
        try:
            p = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout_s)
            out = Path(last).read_text(encoding="utf-8") if Path(last).stat().st_size else p.stdout
            return p.returncode, out, (p.stderr[-2000:] if p.returncode != 0 else None)
        finally:
            Path(last).unlink(missing_ok=True)
    cmd = ["agy", "--print", prompt, "--output-format", "text", "--add-dir", wd, "--model", model]
    p = subprocess.run(cmd, cwd=wd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout_s)
    return p.returncode, p.stdout, (p.stderr[-2000:] if p.returncode != 0 else None)


def _call_http(model: str, prompt: str, timeout_s: int) -> tuple[int, str, str | None]:
    """OpenAI 互換 chat/completions（local 経路）。思考出力（reasoning_content / <think>）は捨てて本文だけ返す。"""
    import urllib.request, urllib.error
    key = http_api_key()
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.8, "top_p": 0.95,
                       "max_tokens": HTTP_MAX_TOKENS, "stream": False,
                       "chat_template_kwargs": {"thinking": HTTP_THINKING, "enable_thinking": HTTP_THINKING}}).encode("utf-8")
    req = urllib.request.Request(HTTP_URL, data=body, headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, "", f"http {e.code}: {e.read()[:500]!r}"
    except Exception as e:  # noqa: BLE001
        return 1, "", f"http error: {e!r}"[:500]
    try:
        msg = data["choices"][0]["message"]; text = msg.get("content") or ""
    except (KeyError, IndexError, TypeError):
        return 1, json.dumps(data)[:2000], "unexpected response"
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return 0, text, None


def call_llm(route: str, prompt: str, log_dir: str | os.PathLike, job_id: str, timeout_s: int = 600,
             max_attempts: int = 2, workdir: str | None = None, required_keys: tuple[str, ...] = ()) -> tuple[dict | list | None, LLMResult]:
    tool, model = ROUTES[route]
    log_dir = Path(log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    psha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    (log_dir / f"{job_id}.prompt.md").write_text(prompt, encoding="utf-8")
    started = datetime.datetime.now().astimezone().isoformat()
    t0 = time.perf_counter()
    out_text = ""; rc = -1; err = None; parsed = None; attempts = 0
    # 生成担当には空のディレクトリを作業場所として渡す（リポジトリや他データを参照させない）
    wd = workdir or tempfile.mkdtemp(prefix="toramon-llm-")
    for attempts in range(1, max_attempts + 1):
        if tool == "http":
            rc, out_text, err = _call_http(model, prompt, timeout_s)
        else:
            try:
                rc, out_text, cli_err = _run_cli(tool, model, prompt, timeout_s, wd)
                if rc != 0 and not out_text.strip():
                    err = cli_err
            except subprocess.TimeoutExpired:
                rc, err = 124, "timeout"
            except FileNotFoundError:
                rc, err = 127, f"{tool} CLI が見つかりません（PATH を確認）"
        parsed = _extract_json(out_text) if out_text else None
        ep = is_error_payload(parsed) or ("AGY_ERROR" in (out_text or "") and out_text.strip()[:200]) or None
        if ep:
            err = f"provider_error: {ep}"; parsed = None
            if "429" in ep or "RESOURCE_EXHAUSTED" in ep or "quota" in ep.lower():
                break  # クォータ枯渇は再試行しない（ジョブ保留）
            continue
        if parsed is not None and required_keys and not (isinstance(parsed, dict) and all(k in parsed for k in required_keys)):
            err = f"missing_keys: {[k for k in required_keys if not (isinstance(parsed, dict) and k in parsed)]}"; parsed = None; continue
        if parsed is not None:
            break
    elapsed = time.perf_counter() - t0
    raw_path = log_dir / f"{job_id}.raw.txt"
    raw_path.write_text(out_text, encoding="utf-8")
    res = LLMResult(route=route, tool=tool, model=model, prompt_sha256=psha, started_at=started, elapsed_s=round(elapsed, 2),
                    exit_code=rc, output_chars=len(out_text), attempts=attempts, json_ok=parsed is not None,
                    raw_path=str(raw_path), error=err)
    (log_dir / f"{job_id}.meta.json").write_text(json.dumps(asdict(res), ensure_ascii=False, indent=2), encoding="utf-8")
    return parsed, res
