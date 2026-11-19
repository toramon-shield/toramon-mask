#!/usr/bin/env python3
"""Toramon Gateway（最小構成の試作、2026-11-11。Anthropic 形式と Codex 対応は 2026-11-18）
社員の端末やお客様の環境で動くローカルプロキシ。クラウド AI へ送る前に個人情報を端末内で伏せ、件数だけを監査ログに残す。
OpenAI 互換（Chat Completions / Responses）と Anthropic Messages（Claude Code）に対応する。

  .venv/bin/python scripts/toramon_gateway.py --gguf runs/gguf/toramon-to-v9s3/toramon-to-v9s3-q8.gguf --port 8787 --upstream https://api.openai.com
  利用側: OPENAI_BASE_URL=http://127.0.0.1:8787/v1 を設定するだけ（API キーは利用者のものをそのまま転送、ゲートウェイは保存しない）

機能:
  POST /v1/chat/completions, /v1/responses, /v1/embeddings … 本文中の文章を伏せてから上流へ転送。応答の伏せ字は元の値に戻す。
  POST /v1/messages（Claude Code、ANTHROPIC_BASE_URL）… 会話（messages）の文章・ツール入出力を伏せる。システムプロンプトとツール定義は伏せない（--mask-system で伏せる）。
  POST …/responses（Codex、ChatGPT ログインのまま使う場合は --upstream https://chatgpt.com）… input の文章・ツール入出力を伏せる。instructions は伏せない。
  ストリーミング応答（SSE）でも、差分の境目で分かれたプレースホルダをつなぎ直して元の値に戻す。思考（thinking / reasoning）は署名があるので触らない。
  画像・PDF の添付（任意）: --vision-url で共用 OCR サーバー（toramon_vision_server.py）を指定すると、文字を読ませて端末で判定し、該当位置を塗って送る。
    確かめられない添付は止める（--vision-on-fail drop で外す）。src/toramon/vision.py
  --block-labels secret  … パスワードや API キーが含まれる要求は 403 で止める（シャドー AI 対策）。
  GET  /audit/summary?days=7 … 期間・ラベル別の件数集計（原文は一切保存しない）。
  GET  /healthz, POST /redact（ローカル伏せ字 API）
  POST /redact {"restorable": true} と POST /restore … --allow-restore-api のときだけ。対応表はメモリに最長 60 秒、1 回戻すと消える（src/toramon/vault.py）
外部通信は上流（--upstream）だけ。監査ログは JSONL（--audit）。
"""
from __future__ import annotations
import argparse, base64, json, sys, time, datetime, threading, http.client, re, collections, urllib.parse, getpass, html, hashlib, hmac, secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toramon.filter import ToramonFilter, Policy  # noqa: E402
from toramon.vault import RestoreVault, is_loopback, restore_request  # noqa: E402
from toramon.vision import VisionError  # noqa: E402

LABEL_TOKEN = {"private_person": "PERSON", "private_address": "ADDRESS", "private_email": "EMAIL", "private_phone": "PHONE",
               "private_date": "DATE", "account_number": "ID", "private_url": "URL", "secret": "SECRET"}
PH = re.compile(r"\[(PERSON|ADDRESS|EMAIL|PHONE|DATE|ID|URL|SECRET|PII)_(\d+)\]")

ADMIN_HTML = """<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Toramon Gateway 管理画面</title>
<style>
:root{--bg:#f5f6f4;--paper:#fff;--ink:#1d2321;--muted:#5d6865;--line:#d6dbd7;--accent:#1f5f4a;--soft:#e2efe9;--hi:#b4552a}
@media (prefers-color-scheme:dark){:root{--bg:#13181a;--paper:#1b2224;--ink:#e6ebe8;--muted:#a4b0ab;--line:#2f3a37;--accent:#7cc7a8;--soft:#1c322a;--hi:#e0a07c}}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Noto Sans JP","Hiragino Kaku Gothic ProN",sans-serif;font-size:14px;line-height:1.7}
.wrap{max-width:1080px;margin:0 auto;padding:24px 16px}
h1{font-size:20px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 18px;font-size:13px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:18px}
.tile{background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:12px 14px}.tile .k{font-size:12px;color:var(--muted)}.tile .v{font-size:28px;font-weight:700;font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:800px){.grid{grid-template-columns:1fr}}
.card{background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:14px 16px;min-width:0}
h2{font-size:15px;margin:0 0 10px}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}thead th{color:var(--muted);font-weight:600}
.bar{height:10px;background:var(--accent);border-radius:2px;display:inline-block;vertical-align:middle}.blocked{color:var(--hi);font-weight:700}
.ctl{margin-bottom:14px}.ctl select,.ctl button{font:inherit;padding:4px 8px}
.tablewrap{overflow-x:auto}.note{font-size:12px;color:var(--muted);margin-top:14px}
</style></head><body><div class="wrap">
<h1>Toramon Gateway 管理画面</h1><p class="sub">クラウド AI へ送信する前に端末内で伏せた個人情報の件数。原文は保存していません。</p>
<div class="ctl">期間 <select id="days"><option value="1">1 日</option><option value="7" selected>7 日</option><option value="30">30 日</option><option value="90">90 日</option></select> <button id="reload">更新</button> <span id="status" class="sub"></span></div>
<div class="tiles"><div class="tile"><div class="k">要求数</div><div class="v" id="t_req">-</div></div><div class="tile"><div class="k">遮断（秘密情報）</div><div class="v" id="t_blk">-</div></div><div class="tile"><div class="k">伏せた箇所</div><div class="v" id="t_spans">-</div></div><div class="tile"><div class="k">伏せた文字 / 送信文字</div><div class="v" id="t_chars">-</div></div></div>
<div class="grid">
<div class="card"><h2>種類別の件数</h2><div class="tablewrap"><table id="labels"><thead><tr><th>種類</th><th>件数</th><th style="text-align:left">割合</th></tr></thead><tbody></tbody></table></div></div>
<div class="card"><h2>日別の要求数</h2><div class="tablewrap"><table id="days_t"><thead><tr><th>日</th><th>要求</th><th style="text-align:left"></th></tr></thead><tbody></tbody></table></div></div>
<div class="card"><h2>利用者別</h2><div class="tablewrap"><table id="users"><thead><tr><th>利用者</th><th>要求</th><th>遮断</th><th>伏せた箇所</th><th>最終利用</th></tr></thead><tbody></tbody></table></div></div>
<div class="card"><h2>送信先・モデル</h2><div class="tablewrap"><table id="models"><thead><tr><th>送信先 / モデル</th><th>要求</th></tr></thead><tbody></tbody></table></div></div>
</div>
<div class="card" style="margin-top:16px"><h2>最近の要求（原文なし）</h2><div class="tablewrap"><table id="recent"><thead><tr><th>時刻</th><th>利用者</th><th>モデル</th><th>送信文字</th><th>伏せた箇所</th><th>状態</th><th>遅延 ms</th></tr></thead><tbody></tbody></table></div></div>
<p class="note">件数は監査ログ（JSONL）から集計。利用者の帰属は起動オプション --attribute-user（OS ユーザー名 / ヘッダ / 接続元）。</p>
</div>
<script>
const LB={private_person:"氏名",private_address:"住所",private_email:"メール",private_phone:"電話",private_date:"個人の日付",account_number:"識別番号",private_url:"URL",secret:"秘密情報"};
const $=s=>document.querySelector(s);const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
async function load(){const d=$("#days").value;$("#status").textContent="読込中";
 const [s,u,r]=await Promise.all([fetch("/audit/summary?days="+d).then(x=>x.json()),fetch("/audit/by_user?days="+d).then(x=>x.json()),fetch("/audit/recent?days="+d+"&n=50").then(x=>x.json())]);
 $("#t_req").textContent=s.requests;$("#t_blk").textContent=s.blocked;const tot=Object.values(s.masked_spans_by_label).reduce((a,b)=>a+b,0);$("#t_spans").textContent=tot;$("#t_chars").textContent=s.chars_masked+" / "+s.chars_in;
 const lb=Object.entries(s.masked_spans_by_label).sort((a,b)=>b[1]-a[1]);$("#labels tbody").innerHTML=lb.map(([k,v])=>`<tr><td>${esc(LB[k]||k)}</td><td>${v}</td><td style="text-align:left"><span class="bar" style="width:${tot?Math.round(v/tot*160):0}px"></span> ${tot?Math.round(v/tot*100):0}%</td></tr>`).join("")||"<tr><td colspan=3>なし</td></tr>";
 const dm=Math.max(1,...Object.values(s.requests_by_day));$("#days_t tbody").innerHTML=Object.entries(s.requests_by_day).map(([k,v])=>`<tr><td>${esc(k)}</td><td>${v}</td><td style="text-align:left"><span class="bar" style="width:${Math.round(v/dm*160)}px"></span></td></tr>`).join("")||"<tr><td colspan=3>なし</td></tr>";
 $("#users tbody").innerHTML=Object.entries(u.users).map(([k,a])=>`<tr><td>${esc(k)}</td><td>${a.requests}</td><td class="${a.blocked?"blocked":""}">${a.blocked}</td><td>${a.masked_spans}</td><td>${esc(a.last.slice(0,16).replace("T"," "))}</td></tr>`).join("")||"<tr><td colspan=5>なし</td></tr>";
 const mm=[...Object.entries(s.requests_by_upstream).map(([k,v])=>["送信先 "+k,v]),...Object.entries(s.requests_by_model).map(([k,v])=>["モデル "+k,v])];$("#models tbody").innerHTML=mm.map(([k,v])=>`<tr><td>${esc(k)}</td><td>${v}</td></tr>`).join("")||"<tr><td colspan=2>なし</td></tr>";
 $("#recent tbody").innerHTML=r.records.map(x=>`<tr><td>${esc(x.ts.slice(5,16).replace("T"," "))}</td><td>${esc(x.user||"")}</td><td>${esc(x.model||"")}</td><td>${x.chars_in||0}</td><td>${Object.values(x.counts||{}).reduce((a,b)=>a+b,0)}</td><td class="${x.blocked?"blocked":""}">${x.blocked?"遮断":x.status}</td><td>${x.latency_ms||""}</td></tr>`).join("")||"<tr><td colspan=7>なし</td></tr>";
 $("#status").textContent="更新 "+new Date().toLocaleTimeString();}
$("#reload").onclick=load;$("#days").onchange=load;load();setInterval(load,30000);
</script></body></html>"""


class CachedFilter:
    """同じ文章の検出結果を使い回す（エージェントは毎回、会話の履歴全体を送り直すため）。検出器は 1 本ずつ呼ぶ。
    鍵は起動ごとの乱数鍵による文章の HMAC-SHA256、値は位置とラベルだけなので、キャッシュに原文は残らず、短い文章を総当たりで当てることもできない。"""
    def __init__(self, filt: ToramonFilter, size: int = 4096):
        self.filt = filt; self.size = size; self.cache: collections.OrderedDict = collections.OrderedDict(); self.lock = threading.Lock()
        self.model_version, self.rules_version = filt.model_version, filt.rules_version; self._hk = secrets.token_bytes(32)

    def detect(self, text: str):
        key = hmac.new(self._hk, text.encode("utf-8", "surrogatepass"), hashlib.sha256).digest()
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key); r = self.cache[key]
                return type(r)(**{**r.__dict__, "elapsed_ms": 0.0}) if hasattr(r, "__dict__") else r
            r = self.filt.detect(text)
            if getattr(r, "status", "ok") == "ok":
                self.cache[key] = r
                if len(self.cache) > self.size: self.cache.popitem(last=False)
            return r


class Masker:
    """1 要求の中で同じ値には同じプレースホルダを割り当て、応答で元に戻せるようにする。"""
    def __init__(self, filt: ToramonFilter):
        self.filt = filt; self.map: dict[str, str] = {}; self.rev: dict[str, str] = {}; self.counts = collections.Counter(); self.chars_in = 0; self.chars_masked = 0; self.ms = 0.0

    def placeholder(self, val: str, label: str) -> str:
        """値に番号つきプレースホルダを割り当てる（同じ値には同じもの。文章と画像で共通）。"""
        if val not in self.rev:
            tok = LABEL_TOKEN.get(label, "PII"); n = sum(1 for k in self.rev.values() if k.startswith(f"[{tok}_")) + 1
            ph = f"[{tok}_{n}]"; self.rev[val] = ph; self.map[ph] = val
        return self.rev[val]

    def spans(self, text: str, extra_values=(), count: bool = True) -> list[tuple[int, int, str]]:
        """文章の中の個人情報の範囲とプレースホルダの組を返す。extra_values: ほかの検出器が見つけた (値, ラベル)。"""
        if not text: return []
        res = self.filt.detect(text); self.ms += res.elapsed_ms; self.chars_in += len(text) if count else 0
        if res.status != "ok": raise RuntimeError(f"filter error: {res.error}")
        found = [(s["start"], s["end"], s["label"]) for s in res.spans]
        for val, lab in extra_values:
            i = text.find(val) if val else -1
            while i >= 0:
                if not any(a < i + len(val) and i < b for a, b, _ in found): found.append((i, i + len(val), lab))
                i = text.find(val, i + 1)
        out = []
        for st, en, lab in sorted(found):
            out.append((st, en, self.placeholder(text[st:en], lab)))
            if count: self.counts[lab] += 1; self.chars_masked += en - st
        return out

    def mask(self, text: str) -> str:
        if not text: return text
        out = []; i = 0
        for st, en, ph in self.spans(text):
            out.append(text[i:st]); out.append(ph); i = en
        out.append(text[i:]); return "".join(out)

    def restore(self, text: str) -> str:
        return PH.sub(lambda m: self.map.get(m.group(0), m.group(0)), text) if self.map else text

    def restore_json(self, text: str) -> str:
        """JSON の文字列の中身（ツール引数の断片など）に戻すとき。元の値を JSON 用にエスケープする。"""
        return PH.sub(lambda m: json.dumps(self.map[m.group(0)], ensure_ascii=False)[1:-1] if m.group(0) in self.map else m.group(0), text) if self.map else text


# 会話の中で触らないキー（識別子、署名つきの思考、画像などのデータ）
SKIP_KEYS = {"type", "id", "tool_use_id", "call_id", "item_id", "name", "role", "model", "signature", "thinking", "data", "media_type",
             "encrypted_content", "cache_control", "status", "stop_reason", "source", "url", "image_url", "file_id", "citations"}
SKIP_TYPES = {"thinking", "redacted_thinking", "reasoning", "image", "input_image", "document", "input_file"}


def walk_leaves(obj, fn, fn_json=None):
    """会話部分の文字列すべてに fn を適用する。arguments（JSON 文字列）は fn_json。思考・画像などの部品は飛ばす。"""
    if isinstance(obj, dict):
        if obj.get("type") in SKIP_TYPES: return obj
        for k, v in list(obj.items()):
            if k in SKIP_KEYS: continue
            if isinstance(v, str): obj[k] = (fn_json or fn)(v) if k == "arguments" else fn(v)
            elif isinstance(v, (dict, list)): walk_leaves(v, fn, fn_json)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str): obj[i] = fn(v)
            elif isinstance(v, (dict, list)): walk_leaves(v, fn, fn_json)
    return obj


class MediaBlocked(Exception):
    pass


class _SpansFn:
    """ImageMasker に渡す判定関数（その要求の Masker を使うので、画像と文章で番号がそろう）。"""
    def __init__(self, m): self.m = m
    def __call__(self, text, extra=(), count=True): return self.m.spans(text, extra, count)
    def placeholder(self, val, label): self.m.counts[label] += 1; return self.m.placeholder(val, label)


DROP_NOTE = "[添付を外しました: Toramon Gateway が個人情報を確認できなかったため]"


def _data_url(u: str):
    if not (isinstance(u, str) and u.startswith("data:") and ";base64," in u[:120]): return None
    head, b64 = u.split(",", 1); return head[5:].split(";")[0], base64.b64decode(b64)


def mask_media(kind: str, obj, m, gw) -> dict:
    """画像・PDF の添付を、gw.attachments（pass / mask / drop / block）に従って処理する。返り値は件数。
    mask で失敗したとき・中身を確かめられない添付（URL やファイル ID だけのもの）は gw.vision_on_fail（block / drop）に従う。"""
    st = collections.Counter()
    if gw.attachments == "pass": return st
    text_type = "input_text" if kind == "responses" else "text"

    def fail(d, why):
        if gw.attachments == "block" or (gw.attachments == "mask" and gw.vision_on_fail == "block"):
            st["blocked"] += 1; raise MediaBlocked(why)
        d.clear(); d.update({"type": text_type, "text": DROP_NOTE}); st["dropped"] += 1

    def convert(d, mt, data):
        """伏せた (media_type, bytes) を返す。drop のときは None。"""
        if gw.attachments in ("drop", "block"): fail(d, f"attachments={gw.attachments}"); return None
        try:
            if mt == "application/pdf": out = ("application/pdf", gw.imasker.mask_pdf(data, _SpansFn(m)))
            elif mt.startswith("image/"): out = ("image/png", gw.imasker.mask_image(data, _SpansFn(m)))
            else: fail(d, f"unsupported attachment type {mt}"); return None
        except VisionError as e: fail(d, str(e)); return None
        st["masked"] += 1; return out

    def walk(x):
        if isinstance(x, list):
            for v in list(x): walk(v)
            return
        if not isinstance(x, dict): return
        t = x.get("type")
        if t in ("image", "document") and isinstance(x.get("source"), dict):          # Anthropic
            src = x["source"]
            if src.get("type") == "base64":
                r = convert(x, src.get("media_type", ""), base64.b64decode(src.get("data", "")))
                if r: src["media_type"], src["data"] = r[0], base64.b64encode(r[1]).decode()
            elif src.get("type") == "text" and isinstance(src.get("data"), str): src["data"] = m.mask(src["data"])
            elif src.get("type") == "content": walk_leaves(src.get("content"), m.mask)
            else: fail(x, f"attachment by {src.get('type')} cannot be inspected")
            for k in ("title", "context"):
                if isinstance(x.get(k), str): x[k] = m.mask(x[k])
            return
        if t == "image_url":                                                          # Chat Completions
            iu = x.get("image_url"); u = iu.get("url") if isinstance(iu, dict) else iu; du = _data_url(u)
            if not du: return fail(x, "image by URL cannot be inspected")
            r = convert(x, *du)
            if r: x["image_url"] = {**(iu if isinstance(iu, dict) else {}), "url": f"data:{r[0]};base64," + base64.b64encode(r[1]).decode()}
            return
        if t == "file" and isinstance(x.get("file"), dict):
            f = x["file"]; du = _data_url(f.get("file_data"))
            if not du: return fail(x, "file by id cannot be inspected")
            r = convert(x, *du)
            if r: f["file_data"] = f"data:{r[0]};base64," + base64.b64encode(r[1]).decode()
            return
        if t in ("input_image", "input_file"):                                       # Responses
            key = "image_url" if t == "input_image" else "file_data"; du = _data_url(x.get(key))
            if not du: return fail(x, f"{t} by URL or file id cannot be inspected")
            r = convert(x, *du)
            if r: x[key] = f"data:{r[0]};base64," + base64.b64encode(r[1]).decode()
            return
        for k, v in x.items():
            if k not in ("signature", "encrypted_content") and isinstance(v, (dict, list)): walk(v)

    walk(obj); return st


def api_kind(path: str) -> str:
    p = urllib.parse.urlsplit(path).path
    if p.endswith("/messages") or p.endswith("/messages/count_tokens"): return "anthropic"
    if p.endswith("/responses"): return "responses"
    return "openai"


def mask_request(kind: str, obj: dict, m: "Masker", mask_system: bool, gw=None) -> dict:
    media = mask_media(kind, obj.get("messages") if kind != "responses" else obj.get("input"), m, gw) if gw is not None else {}
    _mask_text(kind, obj, m, mask_system); return dict(media)


def _mask_text(kind: str, obj: dict, m: "Masker", mask_system: bool):
    if kind == "anthropic":
        msgs = obj.get("messages") or []
        walk_leaves(msgs, m.mask)
        if mask_system and isinstance(obj.get("system"), (str, list)):
            obj["system"] = m.mask(obj["system"]) if isinstance(obj["system"], str) else walk_leaves(obj["system"], m.mask)
    elif kind == "responses":
        if isinstance(obj.get("input"), str): obj["input"] = m.mask(obj["input"])
        elif isinstance(obj.get("input"), list): walk_leaves(obj["input"], m.mask, lambda v: m.mask(v))
        if mask_system and isinstance(obj.get("instructions"), str): obj["instructions"] = m.mask(obj["instructions"])
    else:
        walk_text(obj, m.mask)


class StreamRestorer:
    """SSE の応答で、差分の境目をまたぐプレースホルダを保留してから元に戻す。"""
    TAIL = re.compile(r"\[[A-Z]*(?:_\d*)?$")

    def __init__(self, m: "Masker"):
        self.m = m; self.buf: dict = {}; self.tmpl: dict = {}

    def _push(self, key, text: str, as_json: bool) -> str:
        t = self.buf.pop(key, "") + text
        mt = self.TAIL.search(t)
        if mt and len(t) - mt.start() <= 16: self.buf[key] = t[mt.start():]; t = t[:mt.start()]
        return self.m.restore_json(t) if as_json else self.m.restore(t)

    def _flush(self, key, as_json: bool) -> str:
        t = self.buf.pop(key, ""); return (self.m.restore_json(t) if as_json else self.m.restore(t)) if t else ""

    def event(self, ev: dict) -> list[dict]:
        """1 つのイベントを受け取り、送るべきイベントの列を返す。"""
        t = ev.get("type", "")
        if t == "content_block_delta":                       # Anthropic
            d = ev.get("delta") or {}; key = ("a", ev.get("index"))
            if d.get("type") == "text_delta": d["text"] = self._push(key, d.get("text", ""), False); self.tmpl[key] = ("text", ev)
            elif d.get("type") == "input_json_delta": d["partial_json"] = self._push(key, d.get("partial_json", ""), True); self.tmpl[key] = ("json", ev)
            return [ev]
        if t == "content_block_stop":
            key = ("a", ev.get("index")); out = []
            if key in self.buf and key in self.tmpl:
                kind, tp = self.tmpl[key]; rest = self._flush(key, kind == "json")
                e2 = json.loads(json.dumps(tp)); e2["delta"]["text" if kind == "text" else "partial_json"] = rest; out.append(e2)
            return out + [ev]
        if t.endswith(".delta") and isinstance(ev.get("delta"), str) and ("output_text" in t or "arguments" in t or "refusal" in t):   # Responses
            key = ("r", ev.get("item_id"), ev.get("output_index"), ev.get("content_index")); as_json = "arguments" in t
            ev["delta"] = self._push(key, ev["delta"], as_json); self.tmpl[key] = (as_json, ev); return [ev]
        out = []
        if t.endswith(".done") and ("output_text" in t or "arguments" in t or "refusal" in t):
            key = ("r", ev.get("item_id"), ev.get("output_index"), ev.get("content_index"))
            if key in self.buf and key in self.tmpl:
                as_json, tp = self.tmpl[key]; e2 = json.loads(json.dumps(tp)); e2["delta"] = self._flush(key, as_json)
                if "sequence_number" in e2: e2.pop("sequence_number")
                out.append(e2)
        if t.startswith("response.") or t in ("message_start", "message_delta"):
            walk_leaves(ev, self.m.restore, self.m.restore_json)
        return out + [ev]


def walk_text(obj, fn, keys=("content", "input", "text", "prompt", "instructions")):
    """OpenAI 互換の要求/応答に含まれる文章フィールドへ fn を適用する（文字列はその場で置換）。"""
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if k in keys and isinstance(v, str): obj[k] = fn(v)
            elif isinstance(v, (dict, list)): walk_text(v, fn, keys)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str) and keys == ("content",): obj[i] = fn(v)
            elif isinstance(v, (dict, list)): walk_text(v, fn, keys)
    return obj


class Gateway:
    def __init__(self, filt: ToramonFilter, upstream: str, audit: Path, block_labels: set[str], restore: bool, attribute: str = "os", mask_system: bool = False, dump_dir: str = "", vault: RestoreVault | None = None,
                 attachments: str = "pass", imasker=None, vision_on_fail: str = "block"):
        u = urllib.parse.urlsplit(upstream); self.up_scheme, self.up_host = u.scheme, u.netloc; self.up_prefix = u.path.rstrip("/")
        self.attachments = attachments; self.imasker = imasker; self.vision_on_fail = vision_on_fail
        if attachments == "mask" and imasker is None: raise ValueError("attachments=mask needs an ImageMasker (--vision-url)")
        self.mask_system = mask_system; self.vault = vault; self.dump_dir = Path(dump_dir) if dump_dir else None
        if self.dump_dir: self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.filt = CachedFilter(filt); self.audit = audit; self.block = block_labels; self.restore = restore; self.lock = threading.Lock(); self.attribute = attribute; self.os_user = getpass.getuser()
        audit.parent.mkdir(parents=True, exist_ok=True)

    def log(self, rec: dict):
        rec["ts"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        with self.lock, open(self.audit, "a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def records(self, days: int):
        since = datetime.datetime.now().astimezone() - datetime.timedelta(days=days)
        if not self.audit.exists(): return []
        out = []
        for l in open(self.audit, encoding="utf-8"):
            try: r = json.loads(l)
            except Exception: continue
            if datetime.datetime.fromisoformat(r["ts"]) >= since: out.append(r)
        return out

    def by_user(self, days: int) -> dict:
        agg = {}
        for r in self.records(days):
            u = r.get("user") or r.get("client") or "?"; a = agg.setdefault(u, {"requests": 0, "blocked": 0, "masked_spans": 0, "chars_in": 0, "labels": collections.Counter(), "last": ""})
            a["requests"] += 1; a["blocked"] += int(r.get("blocked", False)); a["masked_spans"] += sum((r.get("counts") or {}).values()); a["chars_in"] += r.get("chars_in", 0); a["labels"].update(r.get("counts") or {}); a["last"] = max(a["last"], r["ts"])
        return {"days": days, "users": {u: {**a, "labels": dict(a["labels"])} for u, a in sorted(agg.items(), key=lambda kv: -kv[1]["requests"])}}

    def summary(self, days: int) -> dict:
        since = datetime.datetime.now().astimezone() - datetime.timedelta(days=days)
        n = blocked = 0; by_label = collections.Counter(); by_day = collections.Counter(); by_model = collections.Counter(); by_host = collections.Counter(); chars_in = chars_masked = 0
        if self.audit.exists():
            for l in open(self.audit, encoding="utf-8"):
                try: r = json.loads(l)
                except Exception: continue
                if datetime.datetime.fromisoformat(r["ts"]) < since: continue
                n += 1; blocked += int(r.get("blocked", False)); by_day[r["ts"][:10]] += 1; by_model[r.get("model") or "?"] += 1; by_host[r.get("upstream") or "?"] += 1
                chars_in += r.get("chars_in", 0); chars_masked += r.get("chars_masked", 0)
                for k, v in (r.get("counts") or {}).items(): by_label[k] += v
        return {"days": days, "requests": n, "blocked": blocked, "chars_in": chars_in, "chars_masked": chars_masked, "masked_spans_by_label": dict(by_label),
                "requests_by_day": dict(sorted(by_day.items())), "requests_by_model": dict(by_model), "requests_by_upstream": dict(by_host), "note": "原文は保存していない（件数のみ）"}

    def forward(self, method: str, path: str, headers: dict, body: bytes | None, stream_to=None):
        conn = (http.client.HTTPSConnection if self.up_scheme == "https" else http.client.HTTPConnection)(self.up_host, timeout=600)
        hop = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "te", "upgrade", "proxy-authorization", "proxy-connection", "accept-encoding", "x-toramon-user"}
        h = {k: v for k, v in headers.items() if k.lower() not in hop}   # 認証やベータ機能のヘッダはそのまま渡す（ゲートウェイは保存しない）
        h["Host"] = self.up_host; h["Accept-Encoding"] = "identity"
        if body is not None: h["Content-Length"] = str(len(body))
        conn.request(method, self.up_prefix + path, body=body, headers=h); resp = conn.getresponse()
        return conn, resp


def make_handler(gw: Gateway):
    class H(BaseHTTPRequestHandler):
        server_version = "ToramonGateway/0.2"; protocol_version = "HTTP/1.1"
        def log_message(self, *a): pass  # 標準出力に要求内容を出さない

        def _json(self, code: int, obj: dict, no_store: bool = False):
            b = json.dumps(obj, ensure_ascii=False).encode(); self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(b)))
            if no_store: self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            u = urllib.parse.urlsplit(self.path)
            if u.path == "/healthz": return self._json(200, {"status": "ok", "model_version": gw.filt.model_version, "rules_version": gw.filt.rules_version, "upstream": f"{gw.up_scheme}://{gw.up_host}"})
            q = urllib.parse.parse_qs(u.query); days = int(q.get("days", ["7"])[0])
            if u.path == "/audit/summary": return self._json(200, gw.summary(days))
            if u.path == "/audit/by_user": return self._json(200, gw.by_user(days))
            if u.path == "/audit/recent":
                n = int(q.get("n", ["50"])[0]); recs = gw.records(days)[-n:][::-1]
                return self._json(200, {"records": [{k: v for k, v in r.items() if k != "client" or gw.attribute == "ip"} for r in recs]})
            if u.path in ("/admin", "/admin/"):
                b = ADMIN_HTML.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); return self.wfile.write(b)
            return self._proxy("GET", None)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0); body = self.rfile.read(n) if n else b""
            if self.path == "/redact":
                m = Masker(gw.filt)
                try:
                    obj = json.loads(body or b"{}")
                    if obj.get("restorable") and gw.vault is None: return self._json(400, {"error": "restore is disabled; start the gateway with --allow-restore-api"})
                    red = m.mask(obj.get("text", "")); out = {"redacted_text": red, "counts": dict(m.counts), "elapsed_ms": round(m.ms, 1)}
                    if obj.get("restorable"): out["restore_token"], out["expires_in"] = gw.vault.put(m.map)
                    return self._json(200, out, no_store=True)
                except Exception as e: return self._json(400, {"error": str(e)})
                finally: m.map.clear(); m.rev.clear()
            if self.path == "/restore":
                if gw.vault is None: return self._json(400, {"error": "restore is disabled; start the gateway with --allow-restore-api"})
                try: obj = json.loads(body or b"{}")
                except Exception: return self._json(400, {"error": "invalid json"})
                code, out = restore_request(gw.vault, obj if isinstance(obj, dict) else {}); return self._json(code, out, no_store=True)
            return self._proxy("POST", body)

        def _proxy(self, method: str, body: bytes | None):
            t0 = time.perf_counter(); rec = {"client": self.client_address[0], "path": self.path, "method": method, "upstream": gw.up_host}
            rec["user"] = self.headers.get("X-Toramon-User") or (gw.os_user if gw.attribute == "os" else self.client_address[0])
            m = Masker(gw.filt)
            try: return self._proxy_masked(method, body, t0, rec, m)
            finally: m.map.clear(); m.rev.clear()   # 応答を返し終えたら対応表を消す

        def _proxy_masked(self, method, body, t0, rec, m):
            media = {}; obj = None; stream = False; kind = api_kind(self.path)
            if body:
                try: obj = json.loads(body)
                except Exception: obj = None
            if isinstance(obj, dict):
                rec["model"] = obj.get("model"); stream = bool(obj.get("stream"))
                try: media = mask_request(kind, obj, m, gw.mask_system, gw)
                except MediaBlocked as e:
                    rec.update(status=403, blocked=True, blocked_labels=["attachment"]); gw.log(rec)
                    return self._json(403, {"error": {"message": f"blocked by Toramon Gateway: an attachment could not be checked for personal data ({e}). Remove it or send text instead.", "type": "toramon_attachment_block"}})
                except Exception as e:
                    rec.update(status=500, error=f"filter: {e}"); gw.log(rec); return self._json(500, {"error": {"message": "toramon filter error", "type": "toramon_filter_error"}})
                body = json.dumps(obj, ensure_ascii=False).encode()
                if gw.dump_dir:   # 検証用: 上流へ実際に送る本文（伏せた後）を保存する
                    (gw.dump_dir / f"{time.time_ns()}_{kind}.json").write_bytes(body)
            rec.update(chars_in=m.chars_in, chars_masked=m.chars_masked, counts=dict(m.counts), filter_ms=round(m.ms, 1))
            if obj is not None and media: rec["attachments"] = media
            hit = [k for k in m.counts if k in gw.block]
            if hit:
                rec.update(blocked=True, status=403, blocked_labels=hit); gw.log(rec)
                return self._json(403, {"error": {"message": f"blocked by Toramon Gateway: request contains {', '.join(hit)} (counts {json.dumps({k: m.counts[k] for k in hit})}). Remove secrets before sending to external AI.", "type": "toramon_policy_block"}})
            try:
                conn, resp = gw.forward(method, self.path, dict(self.headers), body)
            except Exception as e:
                rec.update(status=502, error=f"upstream: {type(e).__name__}"); gw.log(rec); return self._json(502, {"error": {"message": f"upstream unreachable: {type(e).__name__}", "type": "toramon_upstream_error"}})
            self.send_response(resp.status)
            ctype = resp.getheader("Content-Type", "")
            if stream or "text/event-stream" in ctype:
                for k, v in resp.getheaders():
                    if k.lower() not in ("content-length", "transfer-encoding", "connection", "content-encoding"): self.send_header(k, v)
                self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
                sr = StreamRestorer(m); lines: list[bytes] = []

                def send(b: bytes):
                    if b: self.wfile.write(f"{len(b):X}\r\n".encode() + b + b"\r\n"); self.wfile.flush()

                def emit(block: list[bytes]):
                    """SSE の 1 イベント（空行まで）を変換して送る。"""
                    raw = b"".join(block)
                    if not (gw.restore and m.map): return send(raw)
                    txt = raw.decode("utf-8", "replace"); data = [l[5:].lstrip() for l in txt.splitlines() if l.startswith("data:")]
                    try: ev = json.loads("\n".join(data)) if data else None
                    except Exception: ev = None
                    if not isinstance(ev, dict): return send(m.restore(txt).encode())
                    name = next((l[6:].strip() for l in txt.splitlines() if l.startswith("event:")), None)
                    send(b"".join(((f"event: {e.get('type', name)}\n" if name else "") + "data: " + json.dumps(e, ensure_ascii=False) + "\n\n").encode() for e in sr.event(ev)))

                while True:
                    line = resp.readline()
                    if not line: break
                    lines.append(line)
                    if line in (b"\n", b"\r\n"): emit(lines); lines = []
                if lines: emit(lines)
                self.wfile.write(b"0\r\n\r\n"); self.wfile.flush(); rec.update(status=resp.status, streamed=True)
            else:
                data = resp.read()
                if gw.restore and m.map and "json" in ctype:
                    try:
                        o = json.loads(data)
                        o = walk_leaves(o, m.restore, m.restore_json) if kind in ("anthropic", "responses") else walk_text(o, m.restore, keys=("content", "text", "output_text"))
                        data = json.dumps(o, ensure_ascii=False).encode()
                    except Exception: pass
                for k, v in resp.getheaders():
                    if k.lower() not in ("content-length", "transfer-encoding", "connection", "content-encoding"): self.send_header(k, v)
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); rec.update(status=resp.status)
            conn.close(); rec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1); gw.log(rec)
    return H


def build_filter(a) -> ToramonFilter:
    if a.backend == "monju":
        from toramon.adapters.monju_onnx import load_monju
        d = a.model_dir or a.gguf
        det = load_monju(d, threads=a.threads); mv = f"monju:{Path(d).name}"
    elif a.backend == "S":
        from toramon.adapters.sumeshi_pf import SumeshiPF
        det = SumeshiPF(a.gguf, device=a.device); mv = f"S:{Path(a.gguf).name}"
    else:
        from toramon.adapters.sumeshi_pf import SumeshiPFDecoded
        det = SumeshiPFDecoded(a.gguf, device=a.device, label_space="identity" if a.backend == "T-O" else "S"); mv = f"{a.backend}:{Path(a.gguf).name}:toramon-decode"
    return ToramonFilter(det, mv, policy=Policy(use_rules=True, use_boundary_fix=True, use_keep_suppression=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="S", choices=["monju", "T-O", "T-S", "S"]); ap.add_argument("--model-dir", help="backend monju: monju.json と *.onnx を置いたフォルダ"); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--gguf", default="models/sumeshi-privacy-filter-jp-GGUF/privacy-filter-jp-f16.gguf"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8787); ap.add_argument("--upstream", default="https://api.openai.com")
    ap.add_argument("--audit", default=str(Path.home() / ".toramon/gateway_audit.jsonl")); ap.add_argument("--block-labels", default="secret", help="カンマ区切り。該当ラベルを含む要求は 403")
    ap.add_argument("--no-restore", action="store_true", help="応答の伏せ字を元に戻さない")
    ap.add_argument("--mask-system", action="store_true", help="システムプロンプト（Anthropic の system、Responses の instructions）も伏せる。既定は会話だけ")
    ap.add_argument("--allow-restore-api", action="store_true", help="POST /redact の restorable と POST /restore を有効にする（ループバックで待ち受けるときだけ）")
    ap.add_argument("--restore-ttl", type=float, default=60.0, help="復元用の対応表を持つ秒数（1 回戻すか、この秒数のどちらか早い方で消える。上限 60）")
    ap.add_argument("--vision-url", default="", help="画像・PDF の文字を読む共用サーバー（toramon_vision_server.py）の URL。https 必須（自分の PC 上なら http 可）")
    ap.add_argument("--vision-token-file", default="", help="共用サーバーの利用者トークンを書いたファイル（環境変数 TORAMON_VISION_TOKEN でも可）")
    ap.add_argument("--vision-ca", default="", help="共用サーバーの証明書（社内 CA など）")
    ap.add_argument("--vision-vlm", action="store_true", help="高精度オプション: 共用サーバーの画像対応 AI（VLM）にも確認させる（サーバー側で --vlm-url が必要）")
    ap.add_argument("--vision-vlm-required", action="store_true", help="VLM が失敗したら添付を確かめられなかったものとして扱う（既定は OCR だけで塗って送る）")
    ap.add_argument("--attachments", default="", choices=["", "pass", "mask", "drop", "block"],
                    help="画像・PDF の添付: mask=伏せて送る（--vision-url 指定時の既定）、drop=外して送る、block=要求を止める、pass=そのまま送る（--vision-url なしの既定）")
    ap.add_argument("--vision-on-fail", default="block", choices=["block", "drop"], help="mask で確認できなかった添付: block=要求を止める（既定）、drop=外して送る")
    ap.add_argument("--pdf-max-pages", type=int, default=30); ap.add_argument("--vision-cache-ttl", type=float, default=1800.0, help="OCR 結果を（元画像でしか読めない形で）持つ秒数。0 で持たない")
    ap.add_argument("--dump-dir", default="", help="検証用: 上流へ送る本文（伏せた後）をこのフォルダに保存する")
    ap.add_argument("--attribute-user", default="os", choices=["os", "header", "ip"], help="監査ログの利用者の帰属: os=ゲートウェイを動かす OS ユーザー名（端末配布向け）、header=X-Toramon-User、ip=接続元（共用サーバー向け）")
    a = ap.parse_args()
    vault = None
    if a.allow_restore_api:
        if not is_loopback(a.host): raise SystemExit("--allow-restore-api はループバック（127.0.0.1 / ::1 / localhost）で待ち受けるときだけ使えます（元の値がネットワークに流れるため）")
        if not 0 < a.restore_ttl <= 60: raise SystemExit("--restore-ttl は 0 より大きく 60 以下にしてください")
        vault = RestoreVault(a.restore_ttl)
    imasker = None; attachments = a.attachments or ("mask" if a.vision_url else "pass")
    if attachments == "mask":
        if not a.vision_url: raise SystemExit("--attachments mask には --vision-url が必要です")
        from toramon.vision import VisionClient, ImageMasker, SealedCache
        import os
        token = (Path(a.vision_token_file).read_text().strip() if a.vision_token_file else os.environ.get("TORAMON_VISION_TOKEN", ""))
        imasker = ImageMasker(VisionClient(a.vision_url, token, cafile=a.vision_ca or None, vlm=a.vision_vlm), SealedCache(a.vision_cache_ttl), pdf_max_pages=a.pdf_max_pages,
                               vlm_required=a.vision_vlm_required)
    filt = build_filter(a); gw = Gateway(filt, a.upstream, Path(a.audit), {x for x in a.block_labels.split(",") if x}, not a.no_restore, a.attribute_user, a.mask_system, a.dump_dir, vault,
                                        attachments, imasker, a.vision_on_fail)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(gw))
    print(f"Toramon Gateway listening on http://{a.host}:{a.port}  admin=http://{a.host}:{a.port}/admin  upstream={a.upstream}  model={filt.model_version}  block={sorted(gw.block)}  attachments={attachments}  audit={a.audit}", flush=True)
    try: srv.serve_forever()
    except KeyboardInterrupt: pass


if __name__ == "__main__":
    main()
