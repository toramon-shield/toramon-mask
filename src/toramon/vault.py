"""伏せ字を元に戻すための対応表を、短い時間だけメモリに置く（復元 API・CLI 用）。

約束すること:
  - 対応表（プレースホルダと元の値）は呼び出し側に返さない。返すのは推測できない使い捨ての鍵（restore_token）だけ。
  - 対応表は「1 回元に戻したとき」か「期限（既定 60 秒、上限 60 秒）が来たとき」の早い方で消す。
  - 期限切れ・使用済み・存在しない鍵は区別せずに扱い、元に戻さない（文章は伏せたまま返す）。
  - ディスクにもログにも書かない。鍵そのものも保存せず、SHA-256 の値で引く。
  - 件数の上限を超えたら古いものから消す（消えた分は伏せたままになる。漏れる方向には倒れない）。
Python の文字列はメモリ上で確実に消去できないため、「消す」は参照を外して辞書を空にすることを指す。
"""
from __future__ import annotations

import collections
import hashlib
import ipaddress
import re
import secrets
import threading
import time

MAX_TTL = 60.0
PLACEHOLDER = re.compile(r"\[[A-Z]+_\d+\]")


def _key(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class RestoreVault:
    def __init__(self, ttl: float = MAX_TTL, max_entries: int = 1024, clock=time.monotonic, sweep_interval: float = 1.0):
        if not 0 < ttl <= MAX_TTL:
            raise ValueError(f"ttl must be in (0, {MAX_TTL:g}] seconds")
        self.ttl = float(ttl); self.max_entries = max_entries; self.clock = clock
        self._d: collections.OrderedDict[str, tuple[float, dict]] = collections.OrderedDict()
        self._lock = threading.Lock(); self._stop = threading.Event()
        if sweep_interval:
            threading.Thread(target=self._loop, args=(sweep_interval,), daemon=True, name="toramon-vault-sweep").start()

    def _loop(self, interval: float):
        while not self._stop.wait(interval):
            self.sweep()

    def _sweep_locked(self):
        now = self.clock()
        for k in [k for k, (exp, _) in self._d.items() if now >= exp]:
            self._d.pop(k)[1].clear()

    def put(self, mapping: dict[str, str]) -> tuple[str, float]:
        """対応表を預け、鍵と有効秒数を返す。渡した辞書は複製して持つので、呼び出し側は自分の分を消してよい。"""
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._sweep_locked()
            while len(self._d) >= self.max_entries:
                self._d.popitem(last=False)[1][1].clear()
            self._d[_key(token)] = (self.clock() + self.ttl, dict(mapping))
        return token, self.ttl

    def take(self, token) -> dict[str, str] | None:
        """鍵に対応する表を取り出して保管庫から消す（使い捨て）。期限切れ・不明なら None。使い終えたら呼び出し側で clear() する。"""
        if not isinstance(token, str) or not token:
            return None
        with self._lock:
            item = self._d.pop(_key(token), None)
        if item is None:
            return None
        exp, mapping = item
        if self.clock() >= exp:
            mapping.clear(); return None
        return mapping

    def discard(self, token) -> bool:
        m = self.take(token)
        if m is None: return False
        m.clear(); return True

    def sweep(self):
        with self._lock:
            self._sweep_locked()

    def close(self):
        self._stop.set()
        with self._lock:
            for _, m in self._d.values(): m.clear()
            self._d.clear()

    def __len__(self):
        with self._lock:
            return len(self._d)


def restore_text(text: str, mapping: dict[str, str]) -> tuple[str, int]:
    """対応表にあるプレースホルダだけを元に戻す。戻した数も返す。"""
    n = 0
    def sub(m):
        nonlocal n
        v = mapping.get(m.group(0))
        if v is None: return m.group(0)
        n += 1; return v
    return PLACEHOLDER.sub(sub, text), n


def is_loopback(host: str) -> bool:
    if host in ("localhost", "localhost.localdomain"): return True
    try: return ipaddress.ip_address(host).is_loopback
    except ValueError: return False


def restore_request(vault: RestoreVault, obj: dict) -> tuple[int, dict]:
    """POST /restore の本体。{"restore_token", "text"} で元に戻す。{"restore_token", "discard": true} は戻さずに消すだけ。
    鍵が使えないときは 410 と伏せたままの文章を返す（呼び出し側が状態を見なくても、漏れる方向には倒れない）。"""
    token = obj.get("restore_token"); text = obj.get("text", "")
    if obj.get("discard"):
        vault.discard(token); return 200, {"discarded": True}
    if not isinstance(text, str):
        return 400, {"error": "text must be a string"}
    mapping = vault.take(token)
    if mapping is None:
        return 410, {"text": text, "restored": False, "reason": "expired_or_used", "message": "復元の期限切れか使用済みのため、伏せたまま返します"}
    try:
        out, n = restore_text(text, mapping)
    finally:
        mapping.clear()
    return 200, {"text": out, "restored": True, "replaced": n}
