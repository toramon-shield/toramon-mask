"""テスト用の PDF を pdfium で作る（文字情報つきの日本語ページと、画像だけのページ）。"""
import ctypes
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as raw

FONT = next((p for p in ("/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf", "/usr/share/fonts/opentype/ipafont-mincho/ipamp.ttf",
                          "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf") if Path(p).exists()), None)


def make_pdf(path, text_lines=(), image=None, size=(595, 842)):
    """text_lines: 文字情報として置く行。image: PIL 画像（ページ下半分に貼る＝スキャンした紙の代わり）。"""
    doc = pdfium.PdfDocument.new(); page = doc.new_page(*size)
    if text_lines:
        data = Path(FONT).read_bytes(); buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        font = raw.FPDFText_LoadFont(doc.raw, buf, len(data), raw.FPDF_FONT_TRUETYPE, True)
        for i, line in enumerate(text_lines):
            obj = raw.FPDFPageObj_CreateTextObj(doc.raw, font, 14.0)
            ws = (line + "\x00").encode("utf-16-le"); raw.FPDFText_SetText(obj, ctypes.cast(ctypes.create_string_buffer(ws), ctypes.POINTER(raw.FPDF_WCHAR)))
            raw.FPDFPageObj_Transform(obj, 1, 0, 0, 1, 50, size[1] - 60 - 24 * i); raw.FPDFPage_InsertObject(page.raw, obj)
    if image is not None:
        img = pdfium.PdfImage.new(doc); bm = pdfium.PdfBitmap.from_pil(image); img.set_bitmap(bm)
        w, h = image.size; sc = (size[0] - 100) / w
        img.set_matrix(pdfium.PdfMatrix().scale(w * sc, h * sc).translate(50, 60)); page.insert_obj(img)
    page.gen_content(); doc.save(path); doc.close()
