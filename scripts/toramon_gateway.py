#!/usr/bin/env python3
"""Toramon Gateway（最小構成の試作、2026-11-11）
社員の端末やお客様の環境で動く OpenAI 互換のローカルプロキシ。クラウド AI へ送る前に個人情報を端末内で伏せ、件数だけを監査ログに残す。

  .venv/bin/python scripts/toramon_gateway.py --gguf runs/gguf/toramon-to-v9s3/toramon-to-v9s3-q8.gguf --port 8787 --upstream https://api.openai.com
  利用側: OPENAI_BASE_URL=http://127.0.0.1:8787/v1 を設定するだけ（API キーは利用者のものをそのまま転送、ゲートウェイは保存しない）

機能:
  POST /v1/chat/completions, /v1/responses, /v1/embeddings … 本文中の文章を伏せてから上流へ転送。応答の伏せ字は元の値に戻す（非ストリーミング時）。
  --block-labels secret  … パスワードや API キーが含まれる要求は 403 で止める（シャドー AI 対策）。
  GET  /audit/summary?days=7 … 期間・ラベル別の件数集計（原文は一切保存しない）。
  GET  /healthz, POST /redact（ローカル伏せ字 API）
外部通信は上流（--upstream）だけ。監査ログは JSONL（--audit）。
"""
from __future__ import annotations
import argparse, json, sys, time, datetime, threading, http.client, re, collections, urllib.parse, getpass, html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toramon.filter import ToramonFilter, Policy  # noqa: E402

LABEL_TOKEN = {"private_person": "PERSON", "private_address": "ADDRESS", "private_email": "EMAIL", "private_phone": "PHONE",
               "private_date": "DATE", "account_number": "ID", "private_url": "URL", "secret": "SECRET"}
PH = re.compile(r"\[(PERSON|ADDRESS|EMAIL|PHONE|DATE|ID|URL|SECRET)_(\d+)\]")

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


class Masker:
    """1 要求の中で同じ値には同じプレースホルダを割り当て、応答で元に戻せるようにする。"""
    def __init__(self, filt: ToramonFilter):
        self.filt = filt; self.map: dict[str, str] = {}; self.rev: dict[str, str] = {}; self.counts = collections.Counter(); self.chars_in = 0; self.chars_masked = 0; self.ms = 0.0

    def mask(self, text: str) -> str:
        if not text: return text
        res = self.filt.detect(text); self.ms += res.elapsed_ms; self.chars_in += len(text)
        if res.status != "ok": raise RuntimeError(f"filter error: {res.error}")
        out = []; i = 0
        for s in sorted(res.spans, key=lambda s: s["start"]):
            val = text[s["start"]:s["end"]]; tok = LABEL_TOKEN.get(s["label"], "PII")
            if val not in self.rev:
                n = sum(1 for k in self.rev.values() if k.startswith(f"[{tok}_")) + 1
                ph = f"[{tok}_{n}]"; self.rev[val] = ph; self.map[ph] = val
            out.append(text[i:s["start"]]); out.append(self.rev[val]); i = s["end"]
            self.counts[s["label"]] += 1; self.chars_masked += s["end"] - s["start"]
        out.append(text[i:]); return "".join(out)

    def restore(self, text: str) -> str:
        return PH.sub(lambda m: self.map.get(m.group(0), m.group(0)), text) if self.map else text


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
    def __init__(self, filt: ToramonFilter, upstream: str, audit: Path, block_labels: set[str], restore: bool, attribute: str = "os"):
        u = urllib.parse.urlsplit(upstream); self.up_scheme, self.up_host = u.scheme, u.netloc; self.up_prefix = u.path.rstrip("/")
        self.filt = filt; self.audit = audit; self.block = block_labels; self.restore = restore; self.lock = threading.Lock(); self.attribute = attribute; self.os_user = getpass.getuser()
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
        conn = (http.client.HTTPSConnection if self.up_scheme == "https" else http.client.HTTPConnection)(self.up_host, timeout=120)
        h = {k: v for k, v in headers.items() if k.lower() in ("authorization", "content-type", "accept", "openai-organization", "openai-project", "x-api-key", "anthropic-version", "user-agent")}
        h["Host"] = self.up_host
        if body is not None: h["Content-Length"] = str(len(body))
        conn.request(method, self.up_prefix + path, body=body, headers=h); resp = conn.getresponse()
        return conn, resp


def make_handler(gw: Gateway):
    class H(BaseHTTPRequestHandler):
        server_version = "ToramonGateway/0.1"
        def log_message(self, *a): pass  # 標準出力に要求内容を出さない

        def _json(self, code: int, obj: dict):
            b = json.dumps(obj, ensure_ascii=False).encode(); self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

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
            if u.path.startswith("/v1/"): return self._proxy("GET", None)
            self._json(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0); body = self.rfile.read(n) if n else b""
            if self.path == "/redact":
                try:
                    obj = json.loads(body or b"{}"); m = Masker(gw.filt); red = m.mask(obj.get("text", ""))
                    return self._json(200, {"redacted_text": red, "counts": dict(m.counts), "elapsed_ms": round(m.ms, 1)})
                except Exception as e: return self._json(400, {"error": str(e)})
            if self.path.startswith("/v1/"): return self._proxy("POST", body)
            self._json(404, {"error": "not found"})

        def _proxy(self, method: str, body: bytes | None):
            t0 = time.perf_counter(); rec = {"client": self.client_address[0], "path": self.path, "method": method, "upstream": gw.up_host}
            rec["user"] = self.headers.get("X-Toramon-User") or (gw.os_user if gw.attribute == "os" else self.client_address[0])
            m = Masker(gw.filt); obj = None; stream = False
            if body:
                try: obj = json.loads(body)
                except Exception: obj = None
            if isinstance(obj, dict):
                rec["model"] = obj.get("model"); stream = bool(obj.get("stream"))
                try: walk_text(obj, m.mask)
                except Exception as e:
                    rec.update(status=500, error=f"filter: {e}"); gw.log(rec); return self._json(500, {"error": {"message": "toramon filter error", "type": "toramon_filter_error"}})
                body = json.dumps(obj, ensure_ascii=False).encode()
            rec.update(chars_in=m.chars_in, chars_masked=m.chars_masked, counts=dict(m.counts), filter_ms=round(m.ms, 1))
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
                    if k.lower() not in ("content-length", "transfer-encoding", "connection"): self.send_header(k, v)
                self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
                while True:
                    chunk = resp.read(4096)
                    if not chunk: break
                    if gw.restore and m.map: chunk = m.restore(chunk.decode("utf-8", "replace")).encode()  # 行をまたぐプレースホルダは戻らないことがある（試作の既知制限）
                    self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                self.wfile.write(b"0\r\n\r\n"); rec.update(status=resp.status, streamed=True)
            else:
                data = resp.read()
                if gw.restore and m.map and "json" in ctype:
                    try: data = json.dumps(walk_text(json.loads(data), m.restore, keys=("content", "text", "output_text")), ensure_ascii=False).encode()
                    except Exception: pass
                for k, v in resp.getheaders():
                    if k.lower() not in ("content-length", "transfer-encoding", "connection", "content-encoding"): self.send_header(k, v)
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); rec.update(status=resp.status)
            conn.close(); rec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1); gw.log(rec)
    return H


def build_filter(a) -> ToramonFilter:
    if a.backend == "S":
        from toramon.adapters.sumeshi_pf import SumeshiPF
        det = SumeshiPF(a.gguf, device=a.device); mv = f"S:{Path(a.gguf).name}"
    else:
        from toramon.adapters.sumeshi_pf import SumeshiPFDecoded
        det = SumeshiPFDecoded(a.gguf, device=a.device, label_space="identity" if a.backend == "T-O" else "S"); mv = f"{a.backend}:{Path(a.gguf).name}:toramon-decode"
    return ToramonFilter(det, mv, policy=Policy(use_rules=True, use_boundary_fix=True, use_keep_suppression=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="S", choices=["T-O", "T-S", "S"]); ap.add_argument("--gguf", default="models/sumeshi-privacy-filter-jp-GGUF/privacy-filter-jp-f16.gguf"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8787); ap.add_argument("--upstream", default="https://api.openai.com")
    ap.add_argument("--audit", default=str(Path.home() / ".toramon/gateway_audit.jsonl")); ap.add_argument("--block-labels", default="secret", help="カンマ区切り。該当ラベルを含む要求は 403")
    ap.add_argument("--no-restore", action="store_true", help="応答の伏せ字を元に戻さない")
    ap.add_argument("--attribute-user", default="os", choices=["os", "header", "ip"], help="監査ログの利用者の帰属: os=ゲートウェイを動かす OS ユーザー名（端末配布向け）、header=X-Toramon-User、ip=接続元（共用サーバー向け）")
    a = ap.parse_args()
    filt = build_filter(a); gw = Gateway(filt, a.upstream, Path(a.audit), {x for x in a.block_labels.split(",") if x}, not a.no_restore, a.attribute_user)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(gw))
    print(f"Toramon Gateway listening on http://{a.host}:{a.port}  admin=http://{a.host}:{a.port}/admin  upstream={a.upstream}  model={filt.model_version}  block={sorted(gw.block)}  audit={a.audit}", flush=True)
    try: srv.serve_forever()
    except KeyboardInterrupt: pass


if __name__ == "__main__":
    main()
