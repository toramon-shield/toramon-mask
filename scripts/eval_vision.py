#!/usr/bin/env python3
"""画像の伏せ字の精度を測る。Monju PII Bench の文書を画像に描き（1 文字ごとの位置が分かる）、OCR → 判定 → 塗る範囲を正解と比べる。

  .venv/bin/python scripts/eval_vision.py --bench ../monju-mask/artifacts/release/bench/test.jsonl --n 100 \\
      --server http://127.0.0.1:8790 --model-dir ../monju-mask/artifacts/release/model --out results/vision/eval.json

条件: clean = きれいな画像、scan = スキャン風（±1 度の傾き、ぼかし、JPEG 画質 55、ノイズ）
指標: 個人情報（MASK）の範囲がすべて塗られた割合（1 文字でも残れば漏れ）、個人情報の文字のうち塗られた割合、
      個人情報でない文字が塗られた割合（塗りすぎ）。比較として、同じ文書の文章を直接 Monju にかけた場合も出す（OCR を通さない上限）。
"""
from __future__ import annotations
import argparse, io, json, math, random, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402
from toramon.filter import ToramonFilter, Policy  # noqa: E402
from toramon.vision import VisionClient, find_regions, lines_from_ocr, PAD  # noqa: E402

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
W, MARGIN, SIZE, LH, MAXH = 1240, 60, 22, 34, 3508   # A4 を 150 dpi で描いた大きさ、10.5pt 相当


def render(text: str):
    """文書を描き、各文字の箱（描けなかった文字は None）を返す。"""
    f = ImageFont.truetype(FONT, SIZE); boxes = [None] * len(text); x, y = MARGIN, MARGIN
    img = Image.new("RGB", (W, MAXH), "white"); d = ImageDraw.Draw(img)
    for i, ch in enumerate(text):
        if ch == "\n": x, y = MARGIN, y + LH; continue
        w = f.getlength(ch)
        if x + w > W - MARGIN: x, y = MARGIN, y + LH
        if y + LH > MAXH - MARGIN: break
        d.text((x, y), ch, font=f, fill="black"); boxes[i] = (x, y, x + w, y + SIZE * 1.2); x += w
    h = min(MAXH, int(y + LH + MARGIN)); return img.crop((0, 0, W, h)), boxes


def scanify(img, boxes, rng):
    ang = rng.uniform(-1, 1); cx, cy = img.size[0] / 2, img.size[1] / 2; t = math.radians(ang)
    out = img.rotate(ang, resample=Image.BICUBIC, fillcolor="white").filter(ImageFilter.GaussianBlur(0.6))
    px = out.load()
    for _ in range(out.size[0] * out.size[1] // 200):
        u, v = rng.randrange(out.size[0]), rng.randrange(out.size[1]); g = rng.randrange(120, 256); px[u, v] = (g, g, g)
    b = io.BytesIO(); out.save(b, "JPEG", quality=55); out = Image.open(io.BytesIO(b.getvalue())).convert("RGB")

    def rot(x, y):   # PIL の rotate（反時計回り、y は下向き）と同じ変換
        dx, dy = x - cx, y - cy; return cx + dx * math.cos(t) + dy * math.sin(t), cy - dx * math.sin(t) + dy * math.cos(t)
    nb = []
    for bx in boxes:
        if bx is None: nb.append(None); continue
        pts = [rot(bx[0], bx[1]), rot(bx[2], bx[1]), rot(bx[0], bx[3]), rot(bx[2], bx[3])]
        nb.append((min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)))
    return out, nb


def covered(box, regions):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(r[0] - PAD <= cx <= r[2] + PAD and r[1] - PAD <= cy <= r[3] + PAD for r, _ in regions)


class Spans:
    def __init__(self, filt): self.filt = filt; self.vals = {}
    def __call__(self, text, extra=(), count=True):
        r = self.filt.detect(text); return [(s["start"], s["end"], "") for s in r.spans]
    def placeholder(self, v, l): return ""


VARIANTS = {"ocr": dict(vlm=False), "vlm_exact": dict(fuzzy=False, use_box=False), "vlm_fuzzy": dict(fuzzy=True, use_box=False), "vlm_fuzzy_box": dict(fuzzy=True, use_box=True)}


def score(doc, boxes, regions_cover):
    ents = [e for e in doc["entities"] if e.get("action", "MASK") == "MASK"]
    pii = set(); ent_ok = ent_n = 0
    for e in ents:
        idx = [i for i in range(e["start"], e["end"]) if boxes[i] is not None and not doc["text"][i].isspace()]
        if not idx: continue
        ent_n += 1; pii.update(idx); ent_ok += all(regions_cover(i) for i in idx)
    other = [i for i, b in enumerate(boxes) if b is not None and i not in pii and not doc["text"][i].isspace()]
    return {"ent_n": ent_n, "ent_ok": ent_ok, "pii_chars": len(pii), "pii_cov": sum(regions_cover(i) for i in pii),
            "other_chars": len(other), "other_cov": sum(regions_cover(i) for i in other)}


def agg(rows):
    s = {k: sum(r[k] for r in rows) for k in rows[0] if k not in ("ms",)}
    out = {"docs": len(rows), "entities": s["ent_n"], "entity_fully_masked": round(s["ent_ok"] / s["ent_n"], 4),
           "pii_char_coverage": round(s["pii_cov"] / s["pii_chars"], 4), "non_pii_char_masked": round(s["other_cov"] / s["other_chars"], 4)}
    if "ms" in rows[0]: ms = sorted(r["ms"] for r in rows); out.update(ms_median=round(ms[len(ms) // 2]), ms_p90=round(ms[int(len(ms) * 0.9)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True); ap.add_argument("--n", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--server", default="http://127.0.0.1:8790"); ap.add_argument("--token", default=""); ap.add_argument("--vlm", action="store_true")
    ap.add_argument("--model-dir", required=True); ap.add_argument("--conditions", default="clean,scan"); ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--out", default="results/vision/eval.json"); ap.add_argument("--save-examples", type=int, default=3)
    a = ap.parse_args()
    docs = [json.loads(l) for l in open(a.bench, encoding="utf-8")]; rng = random.Random(a.seed); docs = rng.sample(docs, min(a.n, len(docs)))
    from toramon.adapters.monju_onnx import load_monju
    filt = ToramonFilter(load_monju(a.model_dir, threads=2), "monju", policy=Policy()); client = VisionClient(a.server, a.token, timeout=300, vlm=a.vlm)
    out = {"bench": a.bench, "n": len(docs), "seed": a.seed, "vlm": a.vlm, "conditions": {}}
    # 比較: 文章を直接判定（OCR なし）
    rows = []
    for d in docs:
        _, boxes = render(d["text"]); sp = filt.detect(d["text"]).spans; cov = set()
        for s in sp: cov.update(range(s["start"], s["end"]))
        rows.append(score(d, boxes, lambda i: i in cov))
    out["conditions"]["text_only"] = agg(rows); print("text_only", out["conditions"]["text_only"], flush=True)
    ex_dir = Path(a.out).parent / "examples"; ex_dir.mkdir(parents=True, exist_ok=True)
    for cond in a.conditions.split(","):
        raw_path = Path(a.out).with_suffix(f".{cond}.raw.jsonl"); raw_f = open(raw_path, "w", encoding="utf-8"); lock = __import__("threading").Lock()

        def one(k):
            d = docs[k]; img, boxes = render(d["text"])
            if cond == "scan": img, boxes = scanify(img, boxes, random.Random(a.seed * 1000 + k))
            b = io.BytesIO(); img.save(b, "PNG"); t0 = time.time(); res = client.ocr(b.getvalue()); ms = (time.time() - t0) * 1000
            with lock: raw_f.write(json.dumps({"id": d["id"], "ms": ms, "vlm": res.get("vlm"), "vlm_error": res.get("vlm_error")}, ensure_ascii=False) + "\n"); raw_f.flush()
            lines = lines_from_ocr(res); rows = {}
            for name, kw in (VARIANTS.items() if a.vlm else [("ocr", {})]):
                regions = find_regions(lines, Spans(filt), [] if kw.get("vlm") is False else res.get("vlm", []), **{x: y for x, y in kw.items() if x != "vlm"})
                rows[name] = score(d, boxes, lambda i: boxes[i] is not None and covered(boxes[i], regions)); rows[name]["ms"] = ms
                if k < a.save_examples:
                    from toramon.vision import paint
                    paint(img.copy(), [(rg, "[PII]") for rg, _ in regions]).save(ex_dir / f"{cond}_{name}_{d['id']}.png")
            return rows
        with ThreadPoolExecutor(a.parallel) as ex: rows = list(ex.map(one, range(len(docs))))
        raw_f.close()
        for name in rows[0]:
            key = cond if name == "ocr" and not a.vlm else f"{cond}/{name}"
            out["conditions"][key] = agg([r[name] for r in rows]); print(key, out["conditions"][key], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
