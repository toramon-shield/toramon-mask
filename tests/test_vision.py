"""画像・PDF の伏せ字（端末側）と共用 OCR サーバーの約束を、モデルなしで確かめる。"""
import base64, importlib.util, io, json, sys, threading, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")
pytest.importorskip("pypdfium2"); pytest.importorskip("cryptography")
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from toramon.vision import ImageMasker, SealedCache, VisionError, lines_from_ocr, find_regions  # noqa: E402
from pdf_helper import make_pdf, FONT  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


gw = _load("gw_vision", "toramon_gateway.py")
vs = _load("vs_vision", "toramon_vision_server.py")

VALUES = (("山田太郎", "private_person"), ("090-1234-5678", "private_phone"), ("佐藤次郎", "private_person"))


class FakeFilt:
    model_version = rules_version = "fake"
    def detect(self, text):
        from toramon.filter import FilterResult
        spans = []
        for val, lab in VALUES:
            i = text.find(val)
            while i >= 0: spans.append({"start": i, "end": i + len(val), "label": lab}); i = text.find(val, i + 1)
        return FilterResult(status="ok", spans=spans, redacted_text=None, elapsed_ms=0.1, model_version="fake", rules_version="fake", policy_version="fake")


def ocr_result(lines, char_w=20, line_h=30):
    """文字幅 20px・行高 30px で左上から並べた、と見なした OCR 結果。"""
    out = []
    for i, t in enumerate(lines):
        y0 = 10 + i * line_h
        chars = [{"t": c, "box": [10 + j * char_w, y0, 10 + (j + 1) * char_w, y0 + 24]} for j, c in enumerate(t) if not c.isspace()]
        out.append({"text": t, "box": [10, y0, 10 + len(t) * char_w, y0 + 24], "chars": chars})
    return {"width": 400, "height": 200, "lines": out}


class FakeClient:
    def __init__(self, res): self.res = res; self.calls = 0
    def ocr(self, png): self.calls += 1; return json.loads(json.dumps(self.res))


def white(w=400, h=200):
    return Image.new("RGB", (w, h), "white")


def png_bytes(img, fmt="PNG", **kw):
    b = io.BytesIO(); img.save(b, fmt, **kw); return b.getvalue()


def spans_fn():
    m = gw.Masker(FakeFilt()); return m, gw._SpansFn(m)


# ---- 位置合わせと塗りつぶし ----------------------------------------------------------------

def test_lines_from_ocr_aligns_chars_and_skips_spaces():
    ln = lines_from_ocr(ocr_result(["宛先 山田太郎 様"]))[0]
    assert ln["boxes"][3] == [70.0, 10, 90.0, 34] and ln["boxes"][2] is None


def test_lines_from_ocr_falls_back_to_even_split_when_misaligned():
    res = ocr_result(["山田太郎"]); res["lines"][0]["chars"] = [{"t": "X", "box": [0, 0, 1, 1]}]
    ln = lines_from_ocr(res)[0]
    assert all(b is not None for b in ln["boxes"]) and ln["boxes"][0][0] == 10


def test_mask_image_paints_only_pii_and_shares_numbering():
    m, fn = spans_fn(); m.mask("山田太郎")          # 文章側で先に [PERSON_1] が割り当てられている
    im = ImageMasker(FakeClient(ocr_result(["担当 佐藤次郎", "副 山田太郎"])))
    out = Image.open(io.BytesIO(im.mask_image(png_bytes(white()), fn))).convert("RGB")
    assert m.map == {"[PERSON_1]": "山田太郎", "[PERSON_2]": "佐藤次郎"}
    assert out.getpixel((10 + 3 * 20 + 1, 12)) == (0, 0, 0)            # 佐藤次郎の左上
    assert out.getpixel((10 + 1, 20)) == (255, 255, 255)                # 「担当」は残る


def test_mask_image_strips_exif():
    img = white(); exif = Image.Exif(); exif[0x010E] = "山田太郎の写真"; exif[0x8825] = {2: (35.0, 41.0, 0.0)}
    im = ImageMasker(FakeClient(ocr_result([])))
    out = im.mask_image(png_bytes(img, "JPEG", exif=exif), spans_fn()[1])
    o = Image.open(io.BytesIO(out)); assert o.format == "PNG" and not o.getexif() and not o.info.get("exif")


def test_vlm_values_are_added():
    res = ocr_result(["連絡は 080-9999-0000 まで"]); res["vlm"] = [{"text": "080-9999-0000", "label": "private_phone"}, {"text": "手書きの名前", "label": "private_person", "box": [300, 150, 380, 180]}]
    m, fn = spans_fn(); out = Image.open(io.BytesIO(ImageMasker(FakeClient(res)).mask_image(png_bytes(white()), fn)))
    assert set(m.map.values()) == {"080-9999-0000", "手書きの名前"} and out.getpixel((320, 160)) == (0, 0, 0)


def test_ocr_failure_raises_and_no_client_raises():
    class Boom:
        def ocr(self, png): raise VisionError("down")
    with pytest.raises(VisionError): ImageMasker(Boom()).mask_image(png_bytes(white()), spans_fn()[1])
    with pytest.raises(VisionError): ImageMasker(None).mask_image(png_bytes(white()), spans_fn()[1])
    with pytest.raises(VisionError): ImageMasker(FakeClient(ocr_result([]))).mask_image(b"not an image", spans_fn()[1])


# ---- PDF ------------------------------------------------------------------------------------

@pytest.mark.skipif(FONT is None, reason="日本語フォントがない")
def test_text_pdf_is_masked_without_server_and_text_layer_is_gone(tmp_path):
    import pypdfium2 as pdfium
    make_pdf(tmp_path / "a.pdf", ["宛先: 山田太郎 様", "電話: 090-1234-5678"])
    m, fn = spans_fn(); out = ImageMasker(None).mask_pdf((tmp_path / "a.pdf").read_bytes(), fn)
    assert m.map == {"[PERSON_1]": "山田太郎", "[PHONE_1]": "090-1234-5678"}
    d = pdfium.PdfDocument(out); assert d[0].get_textpage().get_text_range() == ""      # 文字情報は残らない


def test_scanned_pdf_needs_server(tmp_path):
    make_pdf(tmp_path / "s.pdf", image=white())
    with pytest.raises(VisionError): ImageMasker(None).mask_pdf((tmp_path / "s.pdf").read_bytes(), spans_fn()[1])
    cl = FakeClient(ocr_result(["山田太郎"])); m, fn = spans_fn()
    ImageMasker(cl).mask_pdf((tmp_path / "s.pdf").read_bytes(), fn)
    assert cl.calls == 1 and m.map == {"[PERSON_1]": "山田太郎"}


def test_pdf_page_limit(tmp_path):
    make_pdf(tmp_path / "a.pdf", image=white())
    with pytest.raises(VisionError): ImageMasker(FakeClient(ocr_result([])), pdf_max_pages=0).mask_pdf((tmp_path / "a.pdf").read_bytes(), spans_fn()[1])


# ---- OCR 結果のキャッシュ ----------------------------------------------------------------------

def test_sealed_cache_needs_original_image_and_expires():
    class C:
        t = 0.0
        def __call__(self): return self.t
    c = C(); sc = SealedCache(ttl=10, clock=c); img = b"image-bytes"
    sc.put(img, {"lines": [{"text": "山田太郎"}]})
    raw = repr(list(sc._d.items())).encode()
    assert "山田太郎".encode() not in raw and img not in raw
    assert sc.get(img) == {"lines": [{"text": "山田太郎"}]} and sc.get(b"other") is None
    c.t = 10; assert sc.get(img) is None and not sc._d


def test_cache_avoids_second_ocr_call():
    cl = FakeClient(ocr_result(["山田太郎"])); im = ImageMasker(cl, SealedCache())
    for _ in range(3): im.mask_image(png_bytes(white()), spans_fn()[1])
    assert cl.calls == 1


# ---- ゲートウェイでの添付の扱い --------------------------------------------------------------

class GW:
    def __init__(self, attachments="mask", on_fail="block", res=None):
        self.attachments = attachments; self.vision_on_fail = on_fail
        self.imasker = ImageMasker(FakeClient(res or ocr_result(["山田太郎"])))


def b64(b): return base64.b64encode(b).decode()


def test_anthropic_image_in_tool_result_is_masked_with_text_numbering():
    m = gw.Masker(FakeFilt()); img = png_bytes(white())
    obj = {"messages": [{"role": "user", "content": [{"type": "text", "text": "佐藤次郎と山田太郎の件"},
            {"type": "tool_result", "tool_use_id": "t", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64(img)}}]}]}]}
    st = gw.mask_request("anthropic", obj, m, False, GW())
    src = obj["messages"][0]["content"][1]["content"][0]["source"]
    assert st == {"masked": 1} and src["media_type"] == "image/png" and src["data"] != b64(img)
    assert obj["messages"][0]["content"][0]["text"] == "[PERSON_2]と[PERSON_1]の件"   # 画像で先に山田太郎=PERSON_1
    assert m.map == {"[PERSON_1]": "山田太郎", "[PERSON_2]": "佐藤次郎"}


def test_remote_and_failed_attachments_are_blocked_or_dropped():
    m = gw.Masker(FakeFilt())
    obj = {"messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}]}]}
    with pytest.raises(gw.MediaBlocked): gw.mask_request("anthropic", obj, m, False, GW())
    gw.mask_request("anthropic", obj, m, False, GW(on_fail="drop"))
    assert obj["messages"][0]["content"][0] == {"type": "text", "text": gw.DROP_NOTE}
    obj = {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64," + b64(b"broken")}]}]}
    with pytest.raises(gw.MediaBlocked): gw.mask_request("responses", obj, m, False, GW())


def test_chat_and_responses_attachments():
    m = gw.Masker(FakeFilt()); u = "data:image/png;base64," + b64(png_bytes(white()))
    chat = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": u, "detail": "high"}}]}]}
    assert gw.mask_request("openai", chat, m, False, GW()) == {"masked": 1}
    assert chat["messages"][0]["content"][0]["image_url"]["detail"] == "high" and chat["messages"][0]["content"][0]["image_url"]["url"] != u
    resp = {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": u}, {"type": "input_file", "file_id": "file-1"}]}]}
    with pytest.raises(gw.MediaBlocked): gw.mask_request("responses", resp, m, False, GW())


def test_attachment_policies_pass_drop_block():
    obj = lambda: {"messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(png_bytes(white()))}}]}]}
    o = obj(); assert gw.mask_request("anthropic", o, gw.Masker(FakeFilt()), False, GW("pass")) == {} and o == obj()
    o = obj(); gw.mask_request("anthropic", o, gw.Masker(FakeFilt()), False, GW("drop")); assert o["messages"][0]["content"][0]["type"] == "text"
    with pytest.raises(gw.MediaBlocked): gw.mask_request("anthropic", obj(), gw.Masker(FakeFilt()), False, GW("block"))


def test_gateway_end_to_end_upstream_sees_masked_image(tmp_path):
    seen = {}

    class Up(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_POST(self):
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            b = json.dumps({"content": [{"type": "text", "text": "[PERSON_1] さんの画像を確認しました"}]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    up = ThreadingHTTPServer(("127.0.0.1", 0), Up); threading.Thread(target=up.serve_forever, daemon=True).start()
    g = gw.Gateway(FakeFilt(), f"http://127.0.0.1:{up.server_address[1]}", tmp_path / "a.jsonl", set(), True, attachments="mask", imasker=GW().imasker)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), gw.make_handler(g)); threading.Thread(target=srv.serve_forever, daemon=True).start()
    body = {"model": "x", "messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(png_bytes(white()))}}, {"type": "text", "text": "誰？"}]}]}
    req = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}/v1/messages", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r: out = json.loads(r.read())
    up.shutdown(); srv.shutdown()
    sent = Image.open(io.BytesIO(base64.b64decode(seen["body"]["messages"][0]["content"][0]["source"]["data"]))).convert("RGB")
    assert sent.getpixel((12, 12)) == (0, 0, 0) and out["content"][0]["text"] == "山田太郎 さんの画像を確認しました"
    assert json.loads((tmp_path / "a.jsonl").read_text().splitlines()[-1])["attachments"] == {"masked": 1}


# ---- 共用 OCR サーバー ---------------------------------------------------------------------

@pytest.fixture
def vision_server(tmp_path, monkeypatch):
    tok_file = tmp_path / "tokens.txt"; token = vs.add_user(tok_file, "sato")
    monkeypatch.setattr(vs, "run_ocr", lambda eng, img: [{"text": "山田太郎", "score": 1.0, "box": [0, 0, 80, 20], "chars": []}])
    cfg = type("C", (), dict(tokens=vs.load_tokens(tok_file), no_auth=False, timeout=5, vlm_url="", vlm_model="", pool=vs.OCRPool(1, 0, lambda: object())))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), vs.make_handler(cfg)); threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", token, tok_file
    srv.shutdown()


def _post(url, body, token=None, raw=None):
    req = urllib.request.Request(url, data=raw if raw is not None else json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(req) as r: return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read())


def test_vision_server_auth_and_formats(vision_server, capsys):
    url, token, tok_file = vision_server
    assert token not in tok_file.read_text() and oct(tok_file.stat().st_mode)[-3:] == "600"
    img = b64(png_bytes(white()))
    assert _post(url + "/v1/ocr", {"image": img})[0] == 401
    assert _post(url + "/v1/ocr", {"image": img}, token="tvs_wrong")[0] == 401
    code, out = _post(url + "/v1/ocr", {"image": img}, token=token)
    assert code == 200 and out["lines"][0]["text"] == "山田太郎" and out["width"] == 400
    assert _post(url + "/v1/ocr", {"image": b64(b"%PDF-1.7 not an image")}, token=token)[0] == 415
    log = capsys.readouterr().out
    assert "山田太郎" not in log and img[:40] not in log and '"user": "sato"' in log


def test_vision_client_refuses_plain_http_to_remote():
    from toramon.vision import VisionClient
    with pytest.raises(VisionError): VisionClient("http://10.0.0.5:8790")
    VisionClient("http://127.0.0.1:8790"); VisionClient("https://vision.example.local")


def test_ocr_pool_rejects_when_full():
    import threading as th
    pool = vs.OCRPool(1, 0, lambda: object()); gate = th.Event(); started = th.Event()
    t = th.Thread(target=lambda: pool.run(lambda e: (started.set(), gate.wait()), 5)); t.start(); started.wait()
    with pytest.raises(OverflowError): pool.run(lambda e: None, 1)
    gate.set(); t.join()


def test_vlm_answer_parsing(monkeypatch):
    class R:
        def __init__(self, b): self.b = b
        def read(self): return self.b
        def __enter__(self): return self
        def __exit__(self, *a): pass
    ans = '説明\n[{"text": "山田太郎", "label": "private_person"}, {"text": "株式会社X", "label": "company"}]'
    monkeypatch.setattr(vs.urllib.request, "urlopen", lambda req, timeout: R(json.dumps({"choices": [{"message": {"content": ans}}]}).encode()))
    assert vs.ask_vlm("http://127.0.0.1:8000/v1", "m", b"png", 5) == [{"text": "山田太郎", "label": "private_person"}]


def test_respaced_detects_names_ocr_glued_together():
    from toramon.vision import respaced
    res = ocr_result(["対象者山田太郎に"]); ch = res["lines"][0]["chars"]
    for c in ch[3:]: c["box"][0] += 8; c["box"][2] += 8          # 「者」と「山」の間が空いている
    lines = lines_from_ocr(res); t, idx = respaced(lines[0]["text"], lines[0]["boxes"])
    assert t == "対象者 山田太郎に" and idx[3] == -1 and idx[4] == 3


def test_join_wrapped_lines():
    res = ocr_result(["住所：東京都千代田区丸", "の内1-2-3", "次の行"]); L = res["lines"]
    L[0]["box"][2] = 400; L[1]["box"][2] = 150                      # 1 行目は右端まで、2 行目は途中まで
    lines = lines_from_ocr(res)
    assert [l["text"] for l in lines] == ["住所：東京都千代田区丸の内1-2-3", "次の行"]


def test_ocr_normalize_postal_mark():
    assert lines_from_ocr(ocr_result(["亍100-0005"]))[0]["text"] == "〒100-0005"


def test_approx_find_tolerates_ocr_misreads_and_spaces():
    from toramon.vision import approx_find
    t = "郵便番号：亍801-0682\n住所：福岡県 北九州市"
    assert [(t[a:b], d) for a, b, d in approx_find(t, "〒801-0682", 2)] == [("亍801-0682", 1)]
    assert [t[a:b] for a, b, _ in approx_find(t, "福岡県北九州市", 1)] == ["福岡県 北九州市"]
    assert approx_find(t, "大阪府", 0) == []


def test_vlm_fuzzy_match_masks_misread_and_restores_vlm_reading():
    res = ocr_result(["郵便番号 亍801-0682"]); res["vlm"] = [{"text": "〒801-0682", "label": "private_address"}]
    m, fn = spans_fn(); out = Image.open(io.BytesIO(ImageMasker(FakeClient(res)).mask_image(png_bytes(white()), fn))).convert("RGB")
    assert m.map == {"[ADDRESS_1]": "〒801-0682"} and out.getpixel((10 + 5 * 20 + 2, 12)) == (0, 0, 0)


def test_vlm_failure_falls_back_to_ocr_unless_required():
    res = ocr_result(["山田太郎"]); res["vlm_error"] = "vlm TimeoutError"
    m, fn = spans_fn(); ImageMasker(FakeClient(res), SealedCache()).mask_image(png_bytes(white()), fn)
    assert m.map == {"[PERSON_1]": "山田太郎"}
    with pytest.raises(VisionError): ImageMasker(FakeClient(res), vlm_required=True).mask_image(png_bytes(white()), spans_fn()[1])


def test_vlm_unknown_label_kept_as_pii(monkeypatch):
    class R:
        def __init__(self, b): self.b = b
        def read(self): return self.b
        def __enter__(self): return self
        def __exit__(self, *a): pass
    ans = '[{"text": "山田太郎", "bbox_2d": [100, 200, 300, 250], "label": "private_person|private_address"}]'
    monkeypatch.setattr(vs.urllib.request, "urlopen", lambda req, timeout: R(json.dumps({"choices": [{"message": {"content": ans}}]}).encode()))
    assert vs.ask_vlm("http://127.0.0.1:8000/v1", "m", b"png", 5, (1000, 500)) == [{"text": "山田太郎", "label": "pii", "box": [100.0, 100.0, 300.0, 125.0]}]
    assert gw.PH.fullmatch("[PII_1]")
