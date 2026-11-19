"""画像・PDF の個人情報を伏せる（端末側）。文字の読み取り（OCR）だけを共用サーバーに頼み、判定と塗りつぶしは端末で行う。

流れ:
  画像 → 共用サーバーで OCR（文字と 1 文字ごとの位置）→ 端末の Monju とルールで判定 → 該当位置を黒く塗り、[PERSON_1] などを書き込む
  PDF  → 端末で 1 ページずつ画像にする。文字情報のある PDF は文字と位置を PDF から直接取り、画像を含むページは OCR も使う
         → 塗った画像だけで新しい PDF を作る（元の文字情報・注釈・作成者などの付帯情報・埋め込みファイルは残らない）

約束すること:
  - 対応表（プレースホルダと元の値）は文章と同じ Masker（その要求だけの対応表）に入る。画像の [PERSON_1] と文章の [PERSON_1] は同じ人。
  - 共用サーバーは何も保存しない（サーバー側の約束。scripts/toramon_vision_server.py）。対応表はサーバーに渡らない。
  - 失敗したら例外（VisionError）を投げる。呼び出し側は送らない（block）か添付を外す（drop）。未処理のまま通さない。
  - 出力の画像は画素から作り直すので、撮影場所（GPS）などの付帯情報は残らない。
  - OCR の結果は、元の画像を持っている人にしか読めない形（画像のハッシュから作る鍵で AES-GCM 暗号化）で、短時間だけキャッシュする。
    エージェントは毎回、会話全体（添付を含む）を送り直すため。
"""
from __future__ import annotations

import base64
import collections
import hashlib
import hmac
import io
import json
import os
import secrets
import ssl
import threading
import time
import urllib.request

MAX_PIXELS = 40_000_000
# PDF の文字情報で、区切り記号が別の文字になって出てくるもの（1 文字を 1 文字に置き換えるので位置はずれない）
# OCR がよく取り違える文字（1 文字を 1 文字に置き換える）
OCR_NORMALIZE = str.maketrans({"亍": "〒", "·": "・"})
SPACE_GAP = 0.15   # 文字の間がこの割合（文字の高さに対して）より空いていたら、OCR が落とした空白があったと見なす
PDF_NORMALIZE = str.maketrans({"\xa0": " ", "\xad": "-", "\u2010": "-", "\u2011": "-", "\u2212": "-", "\ufffe": "-", "\x02": "-"})
PAD = 3


class VisionError(RuntimeError):
    pass


# ---- 共用サーバーへの問い合わせ ---------------------------------------------------------

class VisionClient:
    """共用サーバー（toramon_vision_server.py）の /v1/ocr を呼ぶ。トークンは HTTPS で送る。"""

    def __init__(self, url: str, token: str = "", timeout: float = 60.0, cafile: str | None = None, vlm: bool = False):
        self.url = url.rstrip("/"); self.token = token; self.timeout = timeout; self.vlm = vlm
        self.ctx = ssl.create_default_context(cafile=cafile) if url.startswith("https") else None
        if url.startswith("http://") and not _loopback_url(url):
            raise VisionError("共用サーバーへは https で接続してください（画像に個人情報が含まれるため）")

    def ocr(self, png: bytes) -> dict:
        body = json.dumps({"image": base64.b64encode(png).decode(), "vlm": self.vlm}).encode()
        req = urllib.request.Request(self.url + "/v1/ocr", data=body, method="POST",
                                     headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {self.token}"} if self.token else {})})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=self.ctx) as r:
                return json.loads(r.read())
        except Exception as e:  # 中身（画像）は例外メッセージに入れない
            raise VisionError(f"vision server: {type(e).__name__}: {getattr(e, 'code', '')}") from None


def _loopback_url(url: str) -> bool:
    from urllib.parse import urlsplit
    from .vault import is_loopback
    return is_loopback(urlsplit(url).hostname or "")


# ---- OCR 結果のキャッシュ（元の画像を持つ人にしか読めない形） -------------------------------------

class SealedCache:
    """鍵 = 起動ごとの秘密で作った画像の HMAC。中身は「画像の SHA-256 と起動ごとの秘密」から作る鍵で AES-GCM 暗号化する。
    メモリを覗いても、元の画像がなければ中身（OCR で読んだ文字＝個人情報）は読めない。期限（既定 30 分）と件数で消す。"""

    def __init__(self, ttl: float = 1800.0, max_entries: int = 256, clock=time.monotonic):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        self._aes = AESGCM; self.ttl = ttl; self.max = max_entries; self.clock = clock
        self._k1, self._k2 = secrets.token_bytes(32), secrets.token_bytes(32)
        self._d: collections.OrderedDict = collections.OrderedDict(); self._lock = threading.Lock()

    def _ids(self, data: bytes):
        h = hashlib.sha256(data).digest()
        return hmac.new(self._k1, h, hashlib.sha256).digest(), hmac.new(self._k2, h, hashlib.sha256).digest()

    def get(self, data: bytes):
        if self.ttl <= 0: return None
        cid, key = self._ids(data)
        with self._lock:
            it = self._d.get(cid)
            if it is None: return None
            exp, nonce, ct = it
            if self.clock() >= exp: self._d.pop(cid); return None
        try: return json.loads(self._aes(key).decrypt(nonce, ct, cid))
        except Exception: return None

    def put(self, data: bytes, obj) -> None:
        if self.ttl <= 0: return
        cid, key = self._ids(data); nonce = os.urandom(12)
        ct = self._aes(key).encrypt(nonce, json.dumps(obj, ensure_ascii=False).encode(), cid)
        with self._lock:
            now = self.clock()
            for k in [k for k, (e, _, _) in self._d.items() if now >= e]: self._d.pop(k)
            self._d[cid] = (now + self.ttl, nonce, ct); self._d.move_to_end(cid)
            while len(self._d) > self.max: self._d.popitem(last=False)


# ---- 文字の並びと位置 ------------------------------------------------------------------------

def lines_from_ocr(res: dict) -> list[dict]:
    """サーバーの応答を [{"text", "boxes": 文字ごとの [x0,y0,x1,y1]（空白は None）}] にする。
    1 文字ごとの位置が文字列とそろわない行は、行の箱を文字数で等分する（見つけた範囲より狭く塗ることを避けるため、広めに倒す）。"""
    out = []
    for ln in res.get("lines", []):
        text = ln.get("text", ""); x0, y0, x1, y1 = ln["box"]; boxes = [None] * len(text)
        chars = [c for c in ln.get("chars", []) if c.get("t")]
        i = 0; ok = True
        for c in chars:
            t = c["t"]
            while i < len(text) and text[i].isspace(): i += 1
            if text[i:i + len(t)] != t: ok = False; break
            for k in range(len(t)):
                bx = c["box"]; w = (bx[2] - bx[0]) / len(t)
                boxes[i + k] = [bx[0] + w * k, bx[1], bx[0] + w * (k + 1), bx[3]]
            i += len(t)
        if not ok or any(b is None and not text[j].isspace() for j, b in enumerate(boxes)):
            w = (x1 - x0) / max(1, len(text)); boxes = [[x0 + w * j, y0, x0 + w * (j + 1), y1] for j in range(len(text))]
        out.append({"text": text.translate(OCR_NORMALIZE), "boxes": boxes, "box": [x0, y0, x1, y1]})
    return join_wrapped(out)


def join_wrapped(lines: list[dict]) -> list[dict]:
    """右端まで届いた行と次の行を 1 行につなぐ（折り返しで住所やメールアドレスが 2 行に割れると判定できないため）。"""
    if not lines: return lines
    right = max(l["box"][2] for l in lines); left = min(l["box"][0] for l in lines); out = [dict(lines[0])]
    for l in lines[1:]:
        p = out[-1]; h = p["box"][3] - p["box"][1]
        if p["box"][2] >= right - 1.5 * h and 0 < l["box"][1] - p["box"][1] < 1.8 * h and l["box"][0] <= left + h:
            p["text"] += l["text"]; p["boxes"] = p["boxes"] + l["boxes"]; p["box"] = list(l["box"])   # 次の判定は最後の行の位置で行う
        else:
            out.append(dict(l))
    return out


def respaced(text: str, boxes: list) -> tuple[str, list[int]]:
    """文字の間が空いている所に空白を戻した文章と、各文字の元の位置（戻した空白は -1）を返す。"""
    out, idx = [], []
    for j, ch in enumerate(text):
        if j and boxes[j] is not None and boxes[j - 1] is not None and not ch.isspace() and not text[j - 1].isspace():
            a, b = boxes[j - 1], boxes[j]; h = b[3] - b[1]
            if abs(b[1] - a[1]) < 0.5 * h and b[0] - a[2] > SPACE_GAP * h: out.append(" "); idx.append(-1)
        out.append(ch); idx.append(j)
    return "".join(out), idx


def lines_from_pdf_text(page, scale: float) -> list[dict]:
    """文字情報のある PDF ページから、文字と位置（画像に直したときの画素座標）を取る。"""
    tp = page.get_textpage(); n = tp.count_chars(); h = page.get_height()
    text = (tp.get_text_range(0, n) if n else "").translate(PDF_NORMALIZE)
    if len(text) != n:   # 文字数と位置の数が合わないときは使わない（OCR に任せる）
        tp.close(); return []
    lines, cur, cb = [], [], []
    for i, ch in enumerate(text):
        if ch in "\r\n":
            if cur: lines.append({"text": "".join(cur), "boxes": cb}); cur, cb = [], []
            continue
        l, b, r, t = tp.get_charbox(i)
        cur.append(ch); cb.append(None if ch.isspace() or r <= l else [l * scale, (h - t) * scale, r * scale, (h - b) * scale])
    if cur: lines.append({"text": "".join(cur), "boxes": cb})
    tp.close(); return lines


# ---- 判定して塗る ---------------------------------------------------------------------------

def _runs(boxes: list) -> list[list[float]]:
    """1 つの範囲の文字の箱を、行ごとの長方形にまとめる。"""
    out = []
    for b in boxes:
        if b is None: continue
        if out and min(out[-1][3], b[3]) - max(out[-1][1], b[1]) > 0.5 * (b[3] - b[1]) and b[0] >= out[-1][0] - 2:
            r = out[-1]; r[0], r[1], r[2], r[3] = min(r[0], b[0]), min(r[1], b[1]), max(r[2], b[2]), max(r[3], b[3])
        else:
            out.append(list(b))
    return out


def approx_find(text: str, pat: str, k: int) -> list[tuple[int, int, int]]:
    """text の中で pat との編集距離が k 以下の部分を探す（Sellers の方法。空白は両方とも無視する）。(start, end, 距離) の重ならない組を返す。"""
    pat = "".join(pat.split()); idx = [i for i, c in enumerate(text) if not c.isspace()]; t = "".join(text[i] for i in idx)
    m = len(pat)
    if not m or not t: return []
    D = list(range(m + 1)); S = [0] * (m + 1); cand = []
    for j, c in enumerate(t, 1):
        prev_d, prev_s = D[0], S[0]; D[0], S[0] = 0, j
        for i in range(1, m + 1):
            sub = prev_d + (pat[i - 1] != c); dele = D[i] + 1; ins = D[i - 1] + 1
            nd, ns = min((sub, prev_s), (dele, S[i]), (ins, S[i - 1]))
            prev_d, prev_s = D[i], S[i]; D[i], S[i] = nd, ns
        if D[m] <= k and j > S[m]: cand.append((D[m], S[m], j))
    out = []
    for d, a, b in sorted(cand):
        if all(b <= x or a >= y for x, y, _ in out): out.append((a, b, d))
    return [(idx[a], idx[b - 1] + 1, d) for a, b, d in sorted(out, key=lambda x: x[0])]


def find_regions(lines: list[dict], spans_fn, vlm=(), fuzzy: bool = True, use_box: bool = True) -> list[tuple[list[float], str]]:
    """行の文字を 1 つの文章にして spans_fn（Masker.spans）で判定し、塗る長方形とプレースホルダの組を返す。
    vlm: 高精度オプション（VLM）が答えた [{"text", "label", "box"}]。OCR の文字の中から VLM の答えに近い部分を探して塗る
    （fuzzy=False なら完全一致だけ）。見つからなければ VLM が答えた位置を塗る（use_box）。"""
    text = "\n".join(l["text"] for l in lines); boxes = []
    for l in lines: boxes.extend(l["boxes"]); boxes.append(None)
    found = {(s, e): ph for s, e, ph in spans_fn(text)}
    # OCR は語の間の空白を落とすことがあり、空白がないと氏名を見落とす。空白を戻した文章でも判定し、和集合をとる（塗る側に倒す）
    sp_text, idx = respaced(text, boxes)
    if sp_text != text:
        for s, e, ph in spans_fn(sp_text, count=False):
            o = [idx[k] for k in range(s, e) if idx[k] >= 0]
            if o: found.setdefault((o[0], o[-1] + 1), ph)
    regions = []
    for v in vlm:
        val, lab = (v.get("text") or "").strip(), v.get("label", "")
        if not val: continue
        n = len("".join(val.split())); k = 0 if (n <= 3 or not fuzzy) else max(1, n // 4)
        hits = approx_find(text, val, k)
        if hits:
            ph = spans_fn.placeholder(val, lab)
            for s, e, _ in hits: found.setdefault((s, e), ph)
        elif use_box and v.get("box"):
            regions.append((list(v["box"]), spans_fn.placeholder(val, lab)))
    for (s, e), ph in found.items():
        for r in _runs(boxes[s:e]): regions.append((r, ph))
    return regions


def paint(img, regions):
    from PIL import ImageDraw, ImageFont
    d = ImageDraw.Draw(img); W, H = img.size
    for (x0, y0, x1, y1), ph in regions:
        x0, y0, x1, y1 = max(0, x0 - PAD), max(0, y0 - PAD), min(W, x1 + PAD), min(H, y1 + PAD)
        d.rectangle([x0, y0, x1, y1], fill="black")
        size = max(8, int((y1 - y0) * 0.6))
        while size > 8 and ImageFont.load_default(size).getlength(ph) > (x1 - x0):
            size -= 1
        f = ImageFont.load_default(size)
        if f.getlength(ph) <= (x1 - x0) + 1:
            d.text((x0 + 1, y0 + (y1 - y0 - size) / 2), ph, fill="white", font=f)
    return img


def _open_image(data: bytes):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        im = Image.open(io.BytesIO(data)); im.load()
    except Exception as e:
        raise VisionError(f"画像を読めません: {type(e).__name__}") from None
    if getattr(im, "n_frames", 1) > 1:
        raise VisionError("複数フレームの画像（アニメーション）は扱えません")
    im = im.convert("RGB")
    clean = Image.new("RGB", im.size); clean.paste(im)   # 画素だけを写し、付帯情報（EXIF・GPS など）を捨てる
    return clean


def _png(img) -> bytes:
    b = io.BytesIO(); img.save(b, "PNG"); return b.getvalue()


class ImageMasker:
    """画像・PDF を伏せる。spans_fn は (文章, 追加の値) -> [(start, end, placeholder)]（Masker.spans）。"""

    def __init__(self, client: VisionClient | None, cache: SealedCache | None = None, pdf_scale: float = 2.0, pdf_max_pages: int = 30, vlm_required: bool = False):
        self.client = client; self.cache = cache; self.pdf_scale = pdf_scale; self.pdf_max_pages = pdf_max_pages; self.vlm_required = vlm_required

    def _ocr(self, png: bytes) -> dict:
        if self.client is None: raise VisionError("画像の文字を読むサーバー（--vision-url）が設定されていません")
        if self.cache is not None:
            hit = self.cache.get(png)
            if hit is not None: return hit
        res = self.client.ocr(png)
        if not isinstance(res, dict) or "lines" not in res: raise VisionError("vision server: bad response")
        if res.get("vlm_error"):
            if self.vlm_required: raise VisionError(f"vision server: {res['vlm_error']}")
            return res   # VLM が失敗した結果は覚えない（次の要求でもう一度 VLM に聞く）
        if self.cache is not None: self.cache.put(png, res)
        return res

    def mask_image(self, data: bytes, spans_fn) -> bytes:
        img = _open_image(data); png = _png(img); res = self._ocr(png)
        return _png(paint(img, find_regions(lines_from_ocr(res), spans_fn, res.get("vlm", []))))

    def mask_pdf(self, data: bytes, spans_fn) -> bytes:
        import pypdfium2 as pdfium
        import pypdfium2.raw as raw
        try: pdf = pdfium.PdfDocument(data)
        except Exception as e: raise VisionError(f"PDF を読めません: {type(e).__name__}") from None
        try:
            n = len(pdf)
            if n > self.pdf_max_pages: raise VisionError(f"PDF のページ数が上限を超えています（{n} > {self.pdf_max_pages}）")
            pages = []
            for i in range(n):
                page = pdf[i]
                img = page.render(scale=self.pdf_scale).to_pil().convert("RGB")
                lines = lines_from_pdf_text(page, self.pdf_scale)
                has_image = any(True for _ in page.get_objects(filter=[raw.FPDF_PAGEOBJ_IMAGE], max_depth=4))
                regions = find_regions(lines, spans_fn) if lines else []
                if has_image or not lines:   # 画像を含むページと、文字情報のないページ（スキャン）は OCR も使う
                    res = self._ocr(_png(img)); regions += find_regions(lines_from_ocr(res), spans_fn, res.get("vlm", []))
                pages.append(paint(img, regions)); page.close()
        finally:
            pdf.close()
        out = io.BytesIO()
        pages[0].save(out, "PDF", save_all=True, append_images=pages[1:], resolution=72 * self.pdf_scale)
        return out.getvalue()
