#!/usr/bin/env python3
"""Toramon Vision Server（共用サーバー用）。画像を受け取り、文字と 1 文字ごとの位置を返すだけのサーバー。

  .venv/bin/python scripts/toramon_vision_server.py --tokens /etc/toramon/vision_tokens.txt \\
      --host 0.0.0.0 --port 8790 --tls-cert server.crt --tls-key server.key --workers 4
  .venv/bin/python scripts/toramon_vision_server.py --tokens /etc/toramon/vision_tokens.txt --add-user sato   # 利用者のトークンを発行

個人情報かどうかの判定と塗りつぶし、対応表は各自の PC のゲートウェイが行う（src/toramon/vision.py）。このサーバーは:
  - 画像も読んだ文字も保存しない（メモリ上で処理して返すだけ。ログには利用者名・画素数・処理時間・結果コードだけを出す）
  - 利用者ごとのトークン（Authorization: Bearer）がないと使えない。トークンはハッシュ値だけをファイルに置く
  - ループバック以外で待ち受けるときは TLS（--tls-cert / --tls-key）が必須。社内の TLS 終端の内側に置く場合だけ --behind-tls-proxy
  - 受け付けるのは PNG / JPEG / WebP / GIF(1 枚) / BMP / TIFF(1 枚) の画像だけ。サイズ・画素数・同時処理数に上限を設け、混んだら 503 を返す
高精度オプション: --vlm-url に OpenAI 互換の画像対応 AI（vLLM など、同じサーバー内）を指定すると、要求に "vlm": true が
付いているとき、画像の中の個人情報の文字列も答えさせて返す（OCR が読み落とした手書き・写真の中の文字などを拾うため）。
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import io
import json
import queue
import re
import secrets
import ssl
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toramon.vault import is_loopback  # noqa: E402

MAX_BODY = 25 * 1024 * 1024
MAX_PIXELS = 40_000_000
FORMATS = {"PNG", "JPEG", "WEBP", "GIF", "BMP", "TIFF"}
LABELS = {"private_person", "private_address", "private_email", "private_phone", "private_date", "account_number", "private_url", "secret"}


def load_tokens(path: Path) -> dict[str, str]:
    """1 行 1 人: 利用者名:トークンの SHA-256（16 進）。"""
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                user, h = line.strip().split(":", 1); out[h] = user
    return out


def add_user(path: Path, user: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.@-]{1,64}", user): raise SystemExit("利用者名は英数字と _ . @ - だけにしてください")
    token = "tvs_" + secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f: f.write(f"{user}:{hashlib.sha256(token.encode()).hexdigest()}\n")
    path.chmod(0o600); return token


class OCRPool:
    """OCR エンジンを workers 個並べる。待ちが max_wait 件を超えたら断る。"""

    def __init__(self, workers: int, max_wait: int, factory):
        self.q: queue.Queue = queue.Queue()
        for _ in range(workers): self.q.put(factory())
        self.waiting = threading.BoundedSemaphore(workers + max_wait)

    def run(self, fn, timeout: float):
        if not self.waiting.acquire(blocking=False): raise OverflowError("busy")
        try:
            eng = self.q.get(timeout=timeout)
            try: return fn(eng)
            finally: self.q.put(eng)
        except queue.Empty: raise TimeoutError("queue timeout") from None
        finally: self.waiting.release()


def rapidocr_factory(threads: int):
    def make():
        from rapidocr import RapidOCR, LangRec, OCRVersion, ModelType
        return RapidOCR(params={"Rec.lang_type": LangRec.CH, "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.SERVER,
                                "Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.SERVER, "Global.log_level": "error",
                                "EngineConfig.onnxruntime.intra_op_num_threads": threads})
    return make


def run_ocr(eng, img) -> list[dict]:
    """RapidOCR の結果を [{"text", "score", "box": [x0,y0,x1,y1], "chars": [{"t", "box"}]}] にする。"""
    import numpy as np
    r = eng(np.asarray(img)[:, :, ::-1], return_word_box=True, return_single_char_box=True)
    out = []
    if r.boxes is None: return out
    words = r.word_results or [()] * len(r.txts)
    for box, txt, sc, ws in zip(r.boxes.tolist(), r.txts, r.scores, words):
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        chars = [{"t": w[0], "box": [min(p[0] for p in w[2]), min(p[1] for p in w[2]), max(p[0] for p in w[2]), max(p[1] for p in w[2])]} for w in ws if w and w[0]]
        out.append({"text": txt, "score": round(float(sc), 3), "box": [min(xs), min(ys), max(xs), max(ys)], "chars": chars})
    return out


VLM_PROMPT = ("この画像に写っている個人情報をすべて書き出してください。対象: 氏名、住所、電話番号、メールアドレス、個人に紐づく日付（生年月日など）、"
              "口座・会員・社員番号などの識別番号、個人の URL、パスワードや API キー。会社名・製品名・役職は対象外。"
              "label は次のどれか 1 つ: private_person（氏名）、private_address（住所・郵便番号）、private_email、private_phone、private_date、"
              "account_number（識別番号）、private_url、secret（パスワード・キー）。"
              "画像に書かれている文字列のとおりに、JSON の配列だけで答えてください。例: "
              '[{"text": "山田太郎", "bbox_2d": [120, 340, 260, 362], "label": "private_person"}, {"text": "03-1234-5678", "bbox_2d": [120, 380, 330, 402], "label": "private_phone"}]。'
              "bbox_2d は、その文字列が写っている範囲を、画像の幅と高さをそれぞれ 0〜1000 とした座標で示してください。なければ [] と答えてください。")


def ask_vlm(url: str, model: str, png: bytes, timeout: float, size: tuple[int, int] | None = None) -> list[dict]:
    """VLM に個人情報の文字列と位置を答えさせる。位置（0〜1000）は size（幅, 高さ）があれば画素に直して box に入れる。"""
    body = {"model": model, "temperature": 0, "max_tokens": 2048, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}, {"type": "text", "text": VLM_PROMPT}]}]}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r: txt = json.loads(r.read())["choices"][0]["message"]["content"] or ""
    m = re.search(r"\[.*\]", txt, re.S)
    items = json.loads(m.group(0)) if m else []
    out = []
    for i in items:
        if not (isinstance(i, dict) and i.get("text")): continue
        lab = str(i.get("label", ""))
        if lab not in LABELS:
            if not any(x in lab for x in LABELS): continue   # 会社名など、対象外の種類
            lab = "pii"                                       # 種類の一覧を写しただけの答えなど。捨てずに「種類不明」として塗る
        o = {"text": str(i["text"]), "label": lab}; b = i.get("bbox_2d")
        if size and isinstance(b, list) and len(b) == 4 and all(isinstance(x, (int, float)) for x in b):
            W, H = size; o["box"] = [b[0] * W / 1000, b[1] * H / 1000, b[2] * W / 1000, b[3] * H / 1000]
        out.append(o)
    return out


class Server(ThreadingHTTPServer):
    """TLS の握手を、接続ごとのスレッドの中で行う（受付のスレッドで行うと、止まった 1 人が全員を待たせるため）。"""
    daemon_threads = True; ctx: ssl.SSLContext | None = None

    def finish_request(self, request, client_address):
        request.settimeout(30)
        if self.ctx is not None:
            try: request = self.ctx.wrap_socket(request, server_side=True)
            except (ssl.SSLError, OSError): return
        super().finish_request(request, client_address)


def make_handler(cfg):
    class H(BaseHTTPRequestHandler):
        server_version = "ToramonVision/0.1"; protocol_version = "HTTP/1.1"; timeout = 60
        def log_message(self, *a): pass

        def _json(self, code, obj):
            b = json.dumps(obj, ensure_ascii=False).encode(); self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(b)

        def _user(self):
            auth = self.headers.get("Authorization", "")
            if cfg.no_auth: return "local"
            if not auth.startswith("Bearer "): return None
            h = hashlib.sha256(auth[7:].strip().encode()).hexdigest()
            for k, u in cfg.tokens.items():
                if hmac.compare_digest(k, h): return u
            return None

        def do_GET(self):
            if self.path == "/healthz": return self._json(200, {"status": "ok", "vlm": bool(cfg.vlm_url)})
            return self._json(404, {"error": "not found"})

        def do_POST(self):
            t0 = time.perf_counter(); user = self._user(); log = {"user": user, "path": self.path}
            try:
                if user is None: log["status"] = 401; return self._json(401, {"error": "unauthorized"})
                if self.path != "/v1/ocr": log["status"] = 404; return self._json(404, {"error": "not found"})
                n = int(self.headers.get("Content-Length") or 0)
                if n <= 0 or n > MAX_BODY: log["status"] = 413; return self._json(413, {"error": "body too large"})
                obj = json.loads(self.rfile.read(n))
                from PIL import Image
                Image.MAX_IMAGE_PIXELS = MAX_PIXELS
                try:
                    im = Image.open(io.BytesIO(base64.b64decode(obj["image"], validate=True)))
                    if im.format not in FORMATS or getattr(im, "n_frames", 1) > 1: raise ValueError("format")
                    im.load(); im = im.convert("RGB")
                except Exception:
                    log["status"] = 415; return self._json(415, {"error": "unsupported image"})
                log["px"] = im.size[0] * im.size[1]
                try: lines = cfg.pool.run(lambda e: run_ocr(e, im), cfg.timeout)
                except OverflowError: log["status"] = 503; return self._json(503, {"error": "busy"})
                except TimeoutError: log["status"] = 503; return self._json(503, {"error": "timeout"})
                out = {"width": im.size[0], "height": im.size[1], "lines": lines}
                if obj.get("vlm") and cfg.vlm_url:
                    b = io.BytesIO(); im.save(b, "PNG")
                    try: out["vlm"] = ask_vlm(cfg.vlm_url, cfg.vlm_model, b.getvalue(), cfg.timeout, im.size)
                    except Exception as e: log["error"] = out["vlm_error"] = f"vlm {type(e).__name__}"   # OCR の結果は返す。どうするかは端末側が決める
                log["status"] = 200; return self._json(200, out)
            except Exception as e:
                log["status"] = 500; log["error"] = type(e).__name__; return self._json(500, {"error": "internal"})
            finally:
                log["ms"] = round((time.perf_counter() - t0) * 1000, 1)
                print(json.dumps(log, ensure_ascii=False), flush=True)   # 中身（画像・文字）は出さない
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default=str(Path.home() / ".toramon/vision_tokens.txt")); ap.add_argument("--add-user", default="")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--tls-cert", default=""); ap.add_argument("--tls-key", default=""); ap.add_argument("--behind-tls-proxy", action="store_true")
    ap.add_argument("--no-auth", action="store_true", help="試験用: ループバックで待ち受けるときだけ、トークンなしで使えるようにする")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--threads", type=int, default=4, help="OCR 1 本あたりの CPU スレッド数")
    ap.add_argument("--max-wait", type=int, default=40, help="処理待ちの上限件数。超えたら 503"); ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--vlm-url", default="", help="高精度オプション: OpenAI 互換の画像対応 AI の URL（例 http://127.0.0.1:8000/v1）。同じサーバー内に置く")
    ap.add_argument("--vlm-model", default="")
    a = ap.parse_args()
    tok_path = Path(a.tokens)
    if a.add_user:
        print(add_user(tok_path, a.add_user)); print(f"↑ {a.add_user} さんのトークン（再表示できません）。ハッシュ値を {tok_path} に追記しました", file=sys.stderr); return
    if a.vlm_url and not is_loopback(__import__("urllib.parse").parse.urlsplit(a.vlm_url).hostname or ""):
        raise SystemExit("--vlm-url は同じサーバー内（ループバック）に置いてください（画像を外へ出さないため）")
    loop = is_loopback(a.host)
    if not loop and not (a.tls_cert and a.tls_key) and not a.behind_tls_proxy:
        raise SystemExit("ループバック以外で待ち受けるときは --tls-cert と --tls-key が必要です（社内の TLS 終端の内側なら --behind-tls-proxy）")
    if a.no_auth and not loop: raise SystemExit("--no-auth はループバックで待ち受けるときだけ使えます")
    cfg = argparse.Namespace(tokens=load_tokens(tok_path), no_auth=a.no_auth, timeout=a.timeout, vlm_url=a.vlm_url, vlm_model=a.vlm_model,
                             pool=OCRPool(a.workers, a.max_wait, rapidocr_factory(a.threads)))
    if not cfg.tokens and not a.no_auth: raise SystemExit(f"利用者がいません。--add-user 名前 でトークンを発行してください（{tok_path}）")
    srv = Server((a.host, a.port), make_handler(cfg)); scheme = "http"
    if a.tls_cert:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); ctx.minimum_version = ssl.TLSVersion.TLSv1_2; ctx.load_cert_chain(a.tls_cert, a.tls_key)
        srv.ctx = ctx; scheme = "https"
    print(f"Toramon Vision Server {scheme}://{a.host}:{a.port}  users={len(cfg.tokens)}  workers={a.workers}x{a.threads}threads  vlm={a.vlm_url or 'off'}", flush=True)
    try: srv.serve_forever()
    except KeyboardInterrupt: pass


if __name__ == "__main__":
    main()
