"""sumeshi privacy-filter-jp.cpp (libpf.so) の ctypes アダプタ。

- pf_classify は UTF-8 バイト位置と OpenMed 系ラベル名を返す → コードポイント位置と共通 8 カテゴリへ変換する。
- ランタイム不可分の後処理（端トリム・person 分割）は C 側で行われる（無効化不可）。報告では「モデル＋標準後処理」と表記する。
- ラベル写像は configs/label_mapping.yaml の model_S に従う。
"""
from __future__ import annotations

import ctypes
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..spans import Span, byte_span_to_cp, byte_to_cp_table

ROOT = Path(__file__).resolve().parents[3]
from toramon.paths import PFJP_DIR  # noqa: E402
DEFAULT_LIB = PFJP_DIR / "build/cuda-shared/libpf.so"
DEFAULT_MAPPING = ROOT / "configs/label_mapping.yaml"


class _PfEntity(ctypes.Structure):
    _fields_ = [("start", ctypes.c_int32), ("end", ctypes.c_int32), ("score", ctypes.c_float), ("label", ctypes.c_char_p)]


@dataclass
class RawEntity:
    bstart: int
    bend: int
    score: float
    label: str


class SumeshiPF:
    def __init__(self, gguf_path: str, device: str = "cuda", lib_path: str | os.PathLike | None = None,
                 n_threads: int = 0, threshold: float = 0.5, mapping_path: str | os.PathLike | None = None,
                 max_forward_tokens: int | None = None):
        lib_path = Path(lib_path or DEFAULT_LIB)
        # ggml の共有ライブラリは libpf.so と同じツリー（ggml/src）にある。RPATH で解決されるはずだが保険で先に読み込む。
        ggml_dir = lib_path.parent / "ggml/src"
        for name in ("libggml-base.so", "libggml-cpu.so", "libggml-cuda.so", "libggml.so"):
            p = ggml_dir / name
            if p.exists():
                try:
                    ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
        self.lib = ctypes.CDLL(str(lib_path))
        L = self.lib
        L.pf_load.restype = ctypes.c_void_p
        L.pf_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        L.pf_free.argtypes = [ctypes.c_void_p]
        L.pf_last_error.restype = ctypes.c_char_p
        L.pf_last_error.argtypes = [ctypes.c_void_p]
        L.pf_set_window.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        L.pf_classify.restype = ctypes.c_int
        L.pf_classify.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_float,
                                  ctypes.POINTER(ctypes.POINTER(_PfEntity)), ctypes.POINTER(ctypes.c_size_t)]
        L.pf_entities_free.argtypes = [ctypes.POINTER(_PfEntity), ctypes.c_size_t]
        # 低水準 API（トークン列・生ロジット）。GGUF↔HF のパリティ検証で使う。
        L.pf_tokenize.restype = ctypes.c_int
        L.pf_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
                                  ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),
                                  ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.POINTER(ctypes.c_size_t)]
        L.pf_logits.restype = ctypes.c_int
        L.pf_logits.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t,
                                ctypes.POINTER(ctypes.POINTER(ctypes.c_float))]
        L.pf_buf_free.argtypes = [ctypes.c_void_p]
        t0 = time.perf_counter()
        self.ctx = L.pf_load(str(gguf_path).encode(), device.encode(), n_threads)
        err = L.pf_last_error(self.ctx) if self.ctx else b"pf_load returned NULL"
        if err:
            raise RuntimeError(f"pf_load failed: {err.decode(errors='replace')}")
        if max_forward_tokens:
            L.pf_set_window(self.ctx, max_forward_tokens)
        self.load_seconds = time.perf_counter() - t0
        self.device = device
        self.threshold = threshold
        m = yaml.safe_load(open(mapping_path or DEFAULT_MAPPING, encoding="utf-8"))["model_S"]
        self.label_map: dict[str, str] = dict(m["map"])
        self.drop_others: bool = bool(m.get("drop_others", True))
        mg = m.get("merge_adjacent_same_category", {})
        self.merge_enabled = bool(mg.get("enabled", False))
        self.merge_gap = int(mg.get("max_gap_chars", 0))
        self.gguf_path = str(gguf_path)
        # pf_ctx はスレッドセーフではない（複数スレッドから同時に pf_classify を呼ぶと CUDA illegal memory access で落ちる: 2026-11-09 実測）。
        # 1 コンテキストにつき 1 ロックで直列化する。並列スループットが必要なら SumeshiPFPool を使う。
        self._lock = threading.Lock()

    def close(self) -> None:
        if getattr(self, "ctx", None):
            self.lib.pf_free(self.ctx)
            self.ctx = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # --- 生出力（バイト位置、元ラベル） -----------------------------------
    def classify_raw(self, text: str, threshold: float | None = None) -> list[RawEntity]:
        data = text.encode("utf-8")
        ents = ctypes.POINTER(_PfEntity)()
        n = ctypes.c_size_t(0)
        with self._lock:
            rc = self._classify_locked(data, ents, n, threshold)
        if rc != 0:
            err = self.lib.pf_last_error(self.ctx)
            raise RuntimeError(f"pf_classify failed: {err.decode(errors='replace') if err else rc}")
        out = []
        for i in range(n.value):
            e = ents[i]
            out.append(RawEntity(e.start, e.end, float(e.score), e.label.decode()))
        self.lib.pf_entities_free(ents, n)
        return out

    def _classify_locked(self, data, ents, n, threshold):
        return self.lib.pf_classify(self.ctx, data, len(data), float(self.threshold if threshold is None else threshold),
                                  ctypes.byref(ents), ctypes.byref(n))

    # --- 低水準 API（トークン列・生ロジット） ---------------------------------
    def tokenize(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """pf_tokenize: (token ids, [(byte_start, byte_end), ...])。特殊トークンは付かない。"""
        data = text.encode("utf-8")
        ids = ctypes.POINTER(ctypes.c_int32)()
        offs = ctypes.POINTER(ctypes.c_int32)()
        n = ctypes.c_size_t(0)
        rc = self.lib.pf_tokenize(self.ctx, data, len(data), ctypes.byref(ids), ctypes.byref(offs), ctypes.byref(n))
        if rc != 0:
            err = self.lib.pf_last_error(self.ctx)
            raise RuntimeError(f"pf_tokenize failed: {err.decode(errors='replace') if err else rc}")
        try:
            out_ids = [int(ids[i]) for i in range(n.value)]
            out_offs = [(int(offs[2 * i]), int(offs[2 * i + 1])) for i in range(n.value)]
        finally:
            self.lib.pf_buf_free(ids)
            self.lib.pf_buf_free(offs)
        return out_ids, out_offs

    @property
    def n_labels(self) -> int:
        """ラベル数（GGUF メタデータ embedding_length_out）。pf_logits の行幅に必要。"""
        if getattr(self, "_n_labels", None) is None:
            self._n_labels = len(self.labels)
        return self._n_labels

    @property
    def labels(self) -> list[str]:
        """GGUF の classifier.output_labels（ロジット列 → ラベル名）。"""
        if getattr(self, "_labels", None) is None:
            from gguf import GGUFReader  # host venv にもある軽量依存

            r = GGUFReader(self.gguf_path)
            f = r.fields["openai-privacy-filter.classifier.output_labels"]
            self._labels = [bytes(f.parts[i]).decode("utf-8") for i in f.data]
        return self._labels

    def logits(self, ids: list[int]):
        """pf_logits: トークン id 列 → float32 ndarray [n, n_labels]（softmax 前の生ロジット）。"""
        import numpy as np

        n = len(ids)
        if n == 0:
            return np.zeros((0, self.n_labels), dtype=np.float32)
        arr = (ctypes.c_int32 * n)(*ids)
        out = ctypes.POINTER(ctypes.c_float)()
        rc = self.lib.pf_logits(self.ctx, arr, n, ctypes.byref(out))
        if rc != 0:
            err = self.lib.pf_last_error(self.ctx)
            raise RuntimeError(f"pf_logits failed: {err.decode(errors='replace') if err else rc}")
        try:
            buf = np.ctypeslib.as_array(out, shape=(n * self.n_labels,))
            res = np.array(buf, dtype=np.float32, copy=True).reshape(n, self.n_labels)
        finally:
            self.lib.pf_buf_free(out)
        return res

    # --- 共通座標・共通ラベル ------------------------------------------------
    def detect(self, text: str, threshold: float | None = None, merge: bool | None = None) -> list[Span]:
        raws = self.classify_raw(text, threshold)
        table = byte_to_cp_table(text)
        spans: list[Span] = []
        for r in raws:
            label = self.label_map.get(r.label)
            if label is None:
                if self.drop_others:
                    continue
                label = r.label
            cs, ce = byte_span_to_cp(text, r.bstart, r.bend, table)
            spans.append(Span(cs, ce, label, text[cs:ce], "MASK", "", r.score, f"S:{r.label}"))
        spans.sort(key=lambda s: (s.start, s.end))
        if (self.merge_enabled if merge is None else merge):
            spans = self._merge(spans, text)
        return spans

    def _merge(self, spans: list[Span], text: str) -> list[Span]:
        merged: list[Span] = []
        for s in spans:
            if merged:
                prev = merged[-1]
                gap = text[prev.end : s.start]
                if prev.label == s.label and s.start - prev.end <= self.merge_gap and not gap.strip() and s.start >= prev.end:
                    merged[-1] = Span(prev.start, max(prev.end, s.end), prev.label, text[prev.start : max(prev.end, s.end)], "MASK", "",
                                      min(prev.score or 1.0, s.score or 1.0), prev.source + "+" + s.source)
                    continue
            merged.append(s)
        return merged


class SumeshiPFPool:
    """複数の pf_ctx を持ち、スレッドごとに別コンテキストで並列実行する（同時数 N の計測・サービング用）。
    各コンテキストが重みを別々に GPU へ載せる（F16 で約 2.8GB/ctx）。"""

    def __init__(self, gguf_path: str, n_contexts: int = 2, **kw):
        import queue
        self._q: "queue.Queue[SumeshiPF]" = queue.Queue()
        self.members = [SumeshiPF(gguf_path, **kw) for _ in range(n_contexts)]
        for m in self.members:
            self._q.put(m)
        self.load_seconds = sum(m.load_seconds for m in self.members)
        self.threshold = self.members[0].threshold

    def detect(self, text: str, threshold: float | None = None, merge: bool | None = None):
        m = self._q.get()
        try:
            return m.detect(text, threshold, merge)
        finally:
            self._q.put(m)


# ---------------------------------------------------------------------------
# GGUF（libpf）の生ロジットを HF adapter と同じ復号器で復号する経路
# ---------------------------------------------------------------------------
# 背景: libpf の pf_classify は C 側の復号器（Viterbi + 閾値 + 人名分割）と S 用ラベル写像を前提にしており、
# T-O（33 ラベル = 共通 8 カテゴリ）の GGUF ではラベルが写像されず、T-S でも HF adapter（O バイアス付き Viterbi）と
# 復号結果が食い違う（dev_v1 で T-O 0.928→0.621、T-S 0.921→0.886、2026-11-11 実測）。
# ここでは pf_tokenize + pf_logits で生ロジットを取り、hf_token_cls.HFTokenClassifier.detect と同じ手順
# （不可視文字正規化 → O バイアス付き制約 Viterbi → 平均確率の閾値 → ラベル写像 → トリム → 隣接統合）を numpy で再現する。
_ZW_CHARS_NP = set("​‌‍⁠﻿᠎")


def _is_vs(ch: str) -> bool:
    o = ord(ch); return 0xFE00 <= o <= 0xFE0F or 0xE0100 <= o <= 0xE01EF


def _is_inv(ch: str) -> bool:
    return ch in _ZW_CHARS_NP or _is_vs(ch)


class _BIOESNP:
    def __init__(self, labels: list[str]):
        import numpy as np
        n = len(labels); self.n = n; self.tag = ["O"] * n; self.cat = [-1] * n; self.cats: list[str] = []; idx: dict[str, int] = {}; self.o_id = 0
        for i, lb in enumerate(labels):
            if lb == "O" or "-" not in lb:
                self.tag[i] = "O"; self.o_id = i if lb == "O" else self.o_id; continue
            t, c = lb.split("-", 1)
            if t not in "BIES": continue
            if c not in idx: idx[c] = len(self.cats); self.cats.append(c)
            self.tag[i] = t; self.cat[i] = idx[c]
        self.b_of = {c: -1 for c in range(len(self.cats))}; self.i_of = {c: -1 for c in range(len(self.cats))}
        for i in range(n):
            if self.tag[i] == "B": self.b_of[self.cat[i]] = i
            if self.tag[i] == "I": self.i_of[self.cat[i]] = i
        self.closed = np.array([t in ("O", "E", "S") for t in self.tag]); self.fresh = np.array([t in ("O", "B", "S") for t in self.tag])
        self.pred_b = np.array([self.b_of[self.cat[j]] if self.tag[j] in ("I", "E") else -1 for j in range(n)])
        self.pred_i = np.array([self.i_of[self.cat[j]] if self.tag[j] in ("I", "E") else -1 for j in range(n)])
        self.cont = self.pred_b >= 0


def _viterbi_np(lp, tb: _BIOESNP) -> list[int]:
    import numpy as np
    T, C = lp.shape; NEG = -1e30
    prev = np.full(C, NEG, dtype=np.float64); prev[tb.fresh] = lp[0][tb.fresh]
    bp = np.full((T, C), -1, dtype=np.int64)
    pb = np.clip(tb.pred_b, 0, None); pi = np.clip(tb.pred_i, 0, None)
    for t in range(1, T):
        cur = np.full(C, NEG, dtype=np.float64)
        closed_scores = np.where(tb.closed, prev, NEG); arg_closed = int(closed_scores.argmax()); best_closed = closed_scores[arg_closed]
        cur[tb.fresh] = best_closed + lp[t][tb.fresh]; bp[t][tb.fresh] = arg_closed
        if tb.cont.any():
            sb = np.where(tb.pred_b >= 0, prev[pb], NEG); si = np.where(tb.pred_i >= 0, prev[pi], NEG)
            use_i = si > sb; best = np.where(use_i, si, sb); arg = np.where(use_i, tb.pred_i, tb.pred_b)
            cur[tb.cont] = best[tb.cont] + lp[t][tb.cont]; bp[t][tb.cont] = arg[tb.cont]
        prev = cur
    final = np.where(tb.closed, prev, NEG); j = int(final.argmax()); path = [0] * T
    for t in range(T - 1, -1, -1):
        path[t] = j
        if t > 0: j = int(bp[t][j])
    return path


def _spans_from_path_np(path, probs, tb: _BIOESNP):
    out = []; begin = -1; cat = -1; acc = 0.0
    for t, j in enumerate(path):
        tag = tb.tag[j]
        if tag == "S": out.append((t, t, tb.cats[tb.cat[j]], float(probs[t][j]))); begin = -1
        elif tag == "B": begin, cat, acc = t, tb.cat[j], float(probs[t][j])
        elif tag == "I" and begin >= 0 and tb.cat[j] == cat: acc += float(probs[t][j])
        elif tag == "E" and begin >= 0 and tb.cat[j] == cat: acc += float(probs[t][j]); out.append((begin, t, tb.cats[cat], acc / (t - begin + 1))); begin = -1
        else: begin = -1
    return out


class SumeshiPFDecoded(SumeshiPF):
    """GGUF の生ロジットを Toramon 復号器（HF adapter と同一手順）で復号する。label_space は identity（T-O）または S（T-S / S）。"""

    def __init__(self, gguf_path: str, label_space: str = "identity", o_bias: float = 0.0, normalize_invisible: bool = True,
                 max_tokens: int = 4096, halo: int = 1024, merge_adjacent: bool | None = None, **kw):
        super().__init__(gguf_path, **kw)
        self.tb = _BIOESNP(self.labels); self.o_bias = float(o_bias); self.normalize_invisible = normalize_invisible
        self.max_tokens = max_tokens; self.halo = halo; self.label_space = label_space
        if label_space == "identity":
            self.dec_label_map = None; self.dec_drop_others = False; self.dec_merge = False; self.dec_gap = 0
        else:
            self.dec_label_map = self.label_map; self.dec_drop_others = self.drop_others; self.dec_merge = self.merge_enabled; self.dec_gap = self.merge_gap
        if merge_adjacent is not None: self.dec_merge = merge_adjacent

    def token_logprobs(self, text: str):
        import numpy as np
        ids, boffs = self.tokenize(text); n = len(ids)
        if n == 0: return np.zeros((0, self.n_labels), dtype=np.float32), []
        W, halo = self.max_tokens, self.halo
        if n <= W: windows = [(0, n, 0, n)]
        else:
            stride = W - 2 * halo; windows = []; start = 0
            while True:
                wlen = min(W, n - start); lo = 0 if start == 0 else halo; hi = wlen if start + wlen >= n else wlen - halo
                windows.append((start, wlen, lo, hi))
                if start + wlen >= n: break
                start += stride
        out = np.empty((n, self.n_labels), dtype=np.float32)
        for start, wlen, lo, hi in windows:
            with self._lock:
                lg = self.logits(ids[start:start + wlen]).astype(np.float64)
            m = lg.max(-1, keepdims=True); lp = lg - m - np.log(np.exp(lg - m).sum(-1, keepdims=True))
            out[start + lo:start + hi] = lp[lo:hi]
        return out, boffs

    def detect(self, text: str, threshold: float | None = None, merge: bool | None = None) -> list[Span]:
        import numpy as np
        orig_text = text
        keep = [i for i, ch in enumerate(text) if not _is_inv(ch)]
        cmap = None
        if self.normalize_invisible and len(keep) != len(text):
            text = "".join(orig_text[i] for i in keep); cmap = keep
        lp, boffs = self.token_logprobs(text)
        if lp.shape[0] == 0: return []
        lp_dec = lp
        if self.o_bias: lp_dec = lp.copy(); lp_dec[:, self.tb.o_id] -= self.o_bias
        path = _viterbi_np(lp_dec, self.tb); probs = np.exp(lp)
        thr = self.threshold if threshold is None else threshold
        table = byte_to_cp_table(text); spans: list[Span] = []
        for tb_, te_, cat, score in _spans_from_path_np(path, probs, self.tb):
            if score < thr: continue
            label = cat
            if self.dec_label_map is not None:
                label = self.dec_label_map.get(cat)
                if label is None:
                    if self.dec_drop_others: continue
                    label = cat
            cs, ce = byte_span_to_cp(text, boffs[tb_][0], boffs[te_][1], table)
            while cs < ce and (text[cs].isspace() or text[cs] == "="): cs += 1
            while ce > cs and text[ce - 1].isspace(): ce -= 1
            if cs >= ce: continue
            if cmap is not None:
                ocs = cmap[cs]; oce = cmap[ce - 1] + 1
                while oce < len(orig_text) and _is_vs(orig_text[oce]): oce += 1
                cs, ce = ocs, oce
            spans.append(Span(cs, ce, label, orig_text[cs:ce], "MASK", "", score, f"GGUF:{cat}"))
        spans.sort(key=lambda s: (s.start, s.end))
        if (self.dec_merge if merge is None else merge):
            spans = self._merge(spans, orig_text)
        return spans
