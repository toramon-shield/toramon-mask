#!/usr/bin/env python3
"""Toramon Clip（デスクトップ常駐の試作 v0.1、Windows/macOS 向け。この機（Linux、画面なし）では未検証）

Copilot アプリ、Claude Desktop、ChatGPT Desktop など、ベース URL を変えられないデスクトップアプリ向けの対策。
クリップボードを監視し、個人情報を含む文章がコピーされたら、端末内の Toramon Gateway (/redact) で伏せた文章に置き換える。
利用者が AI アプリに貼り付ける時点で、文章はすでに伏せられている。元の文章は本プロセスのメモリにだけ保持し、
小さな常駐ウィンドウの「元に戻す」で復元できる（ファイルには残さない）。

  python toramon_clip.py --gateway http://127.0.0.1:8787 [--interval 0.3] [--min-chars 8]
依存: tkinter（Python 標準、Windows/macOS のインストーラ同梱）。
限界: キー入力で直接打った文章は対象外（ブラウザ拡張または TLS 検査型プロキシで補う）。画像・ファイルは対象外。
"""
from __future__ import annotations
import argparse, json, time, tkinter as tk, urllib.request


def redact(gw: str, text: str) -> tuple[str, dict]:
    req = urllib.request.Request(gw + "/redact", data=json.dumps({"text": text}).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        d = json.loads(r.read().decode()); return d["redacted_text"], d.get("counts", {})


LB = {"private_person": "氏名", "private_address": "住所", "private_email": "メール", "private_phone": "電話", "private_date": "日付", "account_number": "識別番号", "private_url": "URL", "secret": "秘密情報"}


class App:
    def __init__(self, gw: str, interval: float, min_chars: int):
        self.gw, self.interval, self.min_chars = gw, int(interval * 1000), min_chars
        self.root = tk.Tk(); self.root.title("Toramon Clip"); self.root.geometry("360x150"); self.root.attributes("-topmost", True)
        self.status = tk.StringVar(value="監視中（クリップボードの個人情報を伏せます）"); self.last_original = None; self.last_masked = None; self.total = 0
        tk.Label(self.root, textvariable=self.status, wraplength=340, justify="left").pack(padx=10, pady=8, anchor="w")
        f = tk.Frame(self.root); f.pack(pady=4)
        tk.Button(f, text="元に戻す", command=self.restore).pack(side="left", padx=4)
        self.toggle = tk.BooleanVar(value=True); tk.Checkbutton(f, text="監視する", variable=self.toggle).pack(side="left", padx=4)
        self.count = tk.StringVar(value="伏せた箇所: 0"); tk.Label(self.root, textvariable=self.count).pack(anchor="w", padx=10)
        self.root.after(self.interval, self.tick)

    def clipboard(self) -> str | None:
        try: return self.root.clipboard_get()
        except tk.TclError: return None

    def set_clipboard(self, text: str):
        self.root.clipboard_clear(); self.root.clipboard_append(text); self.root.update()

    def tick(self):
        try:
            if self.toggle.get():
                t = self.clipboard()
                if t and len(t) >= self.min_chars and t != self.last_masked and t != self.last_original:
                    masked, counts = redact(self.gw, t)
                    if masked != t:
                        self.last_original, self.last_masked = t, masked; self.set_clipboard(masked)
                        n = sum(counts.values()); self.total += n
                        self.status.set("コピーした文章の " + str(n) + " 箇所を伏せました（" + "、".join(f"{LB.get(k, k)} {v}" for k, v in counts.items()) + "）。貼り付けると伏せ字版が入ります。")
                        self.count.set(f"伏せた箇所: {self.total}")
                    else:
                        self.last_original = t
        except Exception as e:
            self.status.set("Toramon Gateway に接続できません: " + str(e)[:80])
        self.root.after(self.interval, self.tick)

    def restore(self):
        if self.last_original:
            self.set_clipboard(self.last_original); self.status.set("元の文章をクリップボードに戻しました（社外に貼り付けないでください）")

    def run(self): self.root.mainloop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--gateway", default="http://127.0.0.1:8787"); ap.add_argument("--interval", type=float, default=0.3); ap.add_argument("--min-chars", type=int, default=8)
    a = ap.parse_args(); App(a.gateway, a.interval, a.min_chars).run()
