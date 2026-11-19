"""Monju Mask（monju-mask-ja-01）を動かすアダプタ。CPU だけで動く小型モデル 3 つの多数決。

「三人寄れば文殊の知恵」のとおり、見方の違う小さなモデル（専門家）3 つが同じ文を同時に読み、文字ごとに投票する。
氏名と住所は 1 票、それ以外は 2 票で伏せる。重みは Hugging Face の toramon/monju-mask-ja-01（ONNX、合計 約 230 MB）。

    from toramon.adapters.monju_onnx import load_monju
    det = load_monju("models/monju-mask-ja-01", threads=4)   # .detect(text) -> list[Span]

必要なもの: numpy、onnxruntime、tiktoken（GPU 不要、libpf のビルドも不要）。
入力は OpenAI Privacy Filter と同じ語彙（o200k_base）で、ラベルも同じ 8 分類。
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..spans import Span

# ---------------------------------------------------------------------------
# トークン化（o200k_base）と、トークンごとの文字位置
# ---------------------------------------------------------------------------
_ENC = None


def _enc():
    global _ENC
    if _ENC is None:
        import tiktoken
        _ENC = tiktoken.get_encoding("o200k_base")
    return _ENC


def tokenize(text: str) -> tuple[np.ndarray, np.ndarray]:
    """text -> (ids int64 [n], offs int32 [n,2] コードポイント位置の半開区間)。"""
    enc = _enc()
    ids = enc.encode_ordinary(text)
    n = len(ids)
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 2), dtype=np.int32)
    blen = np.fromiter((len(enc.decode_single_token_bytes(t)) for t in ids), dtype=np.int64, count=n)
    bend = np.cumsum(blen); bstart = bend - blen
    if text.isascii():
        offs = np.stack([bstart, bend], 1).astype(np.int32)
    else:
        # 各文字の開始バイト位置。byte2cp[b] = バイト b を含む文字の番号
        clen = np.fromiter((1 if (o := ord(c)) < 0x80 else 2 if o < 0x800 else 3 if o < 0x10000 else 4 for c in text), dtype=np.int64, count=len(text))
        cstart = np.cumsum(clen) - clen
        s = np.searchsorted(cstart, bstart, side="right") - 1
        e = np.searchsorted(cstart, bend - 1, side="right")
        offs = np.stack([s, e], 1).astype(np.int32)
    return np.asarray(ids, dtype=np.int64), offs


# ---------------------------------------------------------------------------
# BIOES の制約付き復号
# ---------------------------------------------------------------------------
NEG = -1e30


class BIOES:
    def __init__(self, labels: list[str]):
        n = len(labels); self.n = n; self.labels = labels
        self.tag = ["O"] * n; self.cat = [-1] * n; self.cats: list[str] = []; idx: dict[str, int] = {}; self.o_id = 0
        for i, lb in enumerate(labels):
            if lb == "O" or "-" not in lb:
                if lb == "O": self.o_id = i
                continue
            t, c = lb.split("-", 1)
            if t not in "BIES": continue
            if c not in idx: idx[c] = len(self.cats); self.cats.append(c)
            self.tag[i] = t; self.cat[i] = idx[c]
        K = len(self.cats)
        col = lambda t: np.array([next(i for i in range(n) if self.tag[i] == t and self.cat[i] == k) for k in range(K)])
        self.B, self.I, self.E, self.S = col("B"), col("I"), col("E"), col("S")
        self.tag_arr = np.array(self.tag); self.cat_arr = np.array(self.cat)


def viterbi_segment(lp: np.ndarray, tb: BIOES) -> np.ndarray:
    """lp [T,C]（対数確率）。区間の前後は閉じた状態とみなす。最良経路のラベル番号 [T] を返す。"""
    T = lp.shape[0]; K = len(tb.cats)
    lo, lb, li, le, ls = lp[:, tb.o_id].astype(np.float64), lp[:, tb.B].astype(np.float64), lp[:, tb.I].astype(np.float64), lp[:, tb.E].astype(np.float64), lp[:, tb.S].astype(np.float64)
    # 状態: closed（直前が O/E/S）の最良値 c、分類ごとの「span 継続中」（直前が B/I）の最良値 o[k]
    c = 0.0; o = np.full(K, NEG)
    bp_closed = np.zeros(T, dtype=np.int64)       # t で閉じた状態になった方法: 0=O, 1+k=S-k, 1+K+k=E-k
    bp_open = np.zeros((T, K), dtype=np.int8)     # t で継続中になった方法: 0=B（新規）, 1=I（継続）
    for t in range(T):
        cand_s = c + ls[t]; cand_e = o + le[t]
        ks, ke = int(cand_s.argmax()), int(cand_e.argmax())
        best = c + lo[t]; how = 0
        if cand_s[ks] > best: best = cand_s[ks]; how = 1 + ks
        if cand_e[ke] > best: best = cand_e[ke]; how = 1 + K + ke
        nb = c + lb[t]; ni = o + li[t]
        use_i = ni > nb
        o = np.where(use_i, ni, nb); bp_open[t] = use_i
        c = best; bp_closed[t] = how
    path = np.empty(T, dtype=np.int64); t = T - 1; state = -1   # -1 = closed、k = 分類 k が継続中
    while t >= 0:
        if state < 0:
            how = bp_closed[t]
            if how == 0: path[t] = tb.o_id
            elif how <= K: path[t] = tb.S[how - 1]
            else: state = how - 1 - K; path[t] = tb.E[state]
        else:
            if bp_open[t, state]: path[t] = tb.I[state]
            else: path[t] = tb.B[state]; state = -1
        t -= 1
    return path


def decode(lp: np.ndarray, tb: BIOES, sure_o: float = 0.999, min_run: int = 2) -> list[tuple[int, int, str, float]]:
    """lp [T,C] -> [(開始トークン, 終了トークン（含む）, 分類, 平均確率)]"""
    T = lp.shape[0]
    if T == 0: return []
    sure = lp[:, tb.o_id] > np.log(sure_o)
    if sure.all(): return []
    # sure が min_run 個以上続く場所を区切りにして、それ以外の区間を復号する
    cut = sure.copy()
    if min_run > 1:
        cut[:] = False; prev = -1
        for i in list(np.flatnonzero(~sure)) + [T]:   # 「確実に O」でないトークンの間隔が min_run 以上なら、その間を区切りにする
            if i - prev - 1 >= min_run: cut[prev + 1:i] = True
            prev = i
    out = []
    idx = np.flatnonzero(~cut)
    if len(idx) == 0: return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]]); ends = np.concatenate([idx[breaks], [idx[-1]]]) + 1
    for s, e in zip(starts, ends):
        seg = lp[s:e]; path = viterbi_segment(seg, tb)
        begin = -1; cat = -1; acc = 0.0
        for t, j in enumerate(path):
            tag = tb.tag[j]
            if tag == "O": begin = -1; continue
            p = float(np.exp(seg[t, j]))
            if tag == "S": out.append((s + t, s + t, tb.cats[tb.cat[j]], p)); begin = -1
            elif tag == "B": begin, cat, acc = t, tb.cat[j], p
            elif tag == "I" and begin >= 0 and tb.cat[j] == cat: acc += p
            elif tag == "E" and begin >= 0 and tb.cat[j] == cat:
                acc += p; out.append((s + begin, s + t, tb.cats[cat], acc / (t - begin + 1))); begin = -1
            else: begin = -1
    return out


# ---------------------------------------------------------------------------
# 推論エンジン
# ---------------------------------------------------------------------------
_ZW = set("​‌‍⁠﻿᠎")


def _is_vs(ch: str) -> bool:
    o = ord(ch); return 0xFE00 <= o <= 0xFE0F or 0xE0100 <= o <= 0xE01EF


@dataclass
class MonjuConfig:
    mode: str = "mean"                 # soft | vote | mean（soft = 3 人の対数確率を平均して 1 回だけ復号する。既定のモデル設定はこれ）
    bias: float = 0.0                  # soft: O（非該当）の対数確率から引く値。大きいほどマスク寄り（漏れが減り、誤検知が増える）
    k: int = 2                         # vote: 伏せるのに必要な票数
    union_labels: tuple[str, ...] = ("private_person", "private_address")  # vote: 1 票で伏せる分類
    threshold: float = 0.5             # span の平均確率の足切り
    threads: int = 4                   # 全体で使う CPU スレッド数
    window: int = 1024                 # これより長い文書は窓に分けて読む（トークン数）
    halo: int = 128                    # 窓の前後に付ける文脈
    weights: tuple[float, ...] = ()    # mean: 専門家ごとの重み（空なら均等）


class MonjuDetector:
    def __init__(self, model_dir: str | Path, cfg: MonjuConfig | None = None, experts: list[str] | None = None):
        import onnxruntime as ort
        self.dir = Path(model_dir); self.cfg = cfg or MonjuConfig()
        meta = json.load(open(self.dir / "monju.json", encoding="utf-8"))
        self.meta = meta; self.labels = [meta["id2label"][str(i)] for i in range(len(meta["id2label"]))]
        self.tb = BIOES(self.labels)
        names = experts or meta["experts"]
        n = len(names)
        # スレッドの配分: 全員に 1 本ずつ配り、余りは「1 スレッドでの所要時間 ÷ 配分済みスレッド数」が最大の専門家へ回す
        cost = [float(meta.get("configs", {}).get(nm, {}).get("cost_ms", 1.0)) for nm in names]; plan = [1] * n
        for _ in range(max(0, self.cfg.threads - n)):
            i = max(range(n), key=lambda j: cost[j] / plan[j]); plan[i] += 1
        self.thread_plan = plan
        self.sessions = []
        for nm, per in zip(names, plan):
            so = ort.SessionOptions(); so.intra_op_num_threads = per; so.inter_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.sessions.append(ort.InferenceSession(str(self.dir / f"{nm}.onnx"), so, providers=["CPUExecutionProvider"]))
        self.names = names
        # 専門家ごとの窓（注意機構を持つ専門家は学習時の長さに合わせて短い窓で読む）
        cfgs = meta.get("configs", {})
        self.windows = [(cfgs.get(nm, {}).get("window", self.cfg.window), cfgs.get(nm, {}).get("halo", self.cfg.halo)) for nm in names]
        self.pool = ThreadPoolExecutor(max_workers=min(n, max(1, self.cfg.threads))) if n > 1 and self.cfg.threads > 1 else None
        w = np.asarray(self.cfg.weights or [1.0] * n, dtype=np.float32); self.w = w / w.sum()
        self.threshold = self.cfg.threshold

    # --- 各専門家の対数確率 ---
    def _run(self, i: int, ids: np.ndarray) -> np.ndarray:
        sess = self.sessions[i]; n = len(ids); W, halo = self.windows[i]
        if n <= W:
            lg = sess.run(None, {"ids": ids[None, :]})[0][0]
        else:
            lg = np.empty((n, len(self.labels)), dtype=np.float32); stride = W - 2 * halo; start = 0
            while True:
                wlen = min(W, n - start); lo = 0 if start == 0 else halo; hi = wlen if start + wlen >= n else wlen - halo
                lg[start + lo:start + hi] = sess.run(None, {"ids": ids[None, start:start + wlen]})[0][0][lo:hi]
                if start + wlen >= n: break
                start += stride
        m = lg.max(-1, keepdims=True); z = lg - m
        return z - np.log(np.exp(z).sum(-1, keepdims=True))

    def expert_logprobs(self, ids: np.ndarray) -> list[np.ndarray]:
        if self.pool is None:
            return [self._run(i, ids) for i in range(len(self.sessions))]
        return list(self.pool.map(lambda i: self._run(i, ids), range(len(self.sessions))))

    def token_logprobs(self, text: str):
        ids, offs = tokenize(text)
        if len(ids) == 0:
            return np.zeros((0, len(self.labels)), dtype=np.float32), offs
        lps = self.expert_logprobs(ids)
        lp = lps[0] if len(lps) == 1 else sum(w * l for w, l in zip(self.w, lps))
        if len(lps) > 1:
            m = lp.max(-1, keepdims=True); z = lp - m; lp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        return lp, offs

    # --- 復号 ---
    def _spans(self, text: str, lp: np.ndarray, offs: np.ndarray, thr: float, src: str) -> list[tuple[int, int, str, float]]:
        out = []
        for tb_, te_, cat, score in decode(lp, self.tb):
            if score < thr: continue
            cs, ce = int(offs[tb_][0]), int(offs[te_][1])
            while cs < ce and (text[cs].isspace() or text[cs] == "="): cs += 1
            while ce > cs and text[ce - 1].isspace(): ce -= 1
            if cs < ce: out.append((cs, ce, cat, score))
        return out

    def _vote(self, n: int, lists: list[list[tuple[int, int, str, float]]]) -> list[tuple[int, int, str, float]]:
        cats = self.tb.cats; ci = {c: i for i, c in enumerate(cats)}
        cnt = np.zeros((len(cats), n), dtype=np.int16)
        for spans in lists:
            for s, e, c, _ in spans: cnt[ci[c], s:e] += 1
        tot = cnt.sum(0); best = cnt.argmax(0)
        need = np.full(len(cats), self.cfg.k); need[[ci[c] for c in self.cfg.union_labels if c in ci]] = 1
        lab = np.where((tot > 0) & (tot >= need[best]), best, -1)
        out = []; idx = np.flatnonzero(lab >= 0)
        if len(idx) == 0: return out
        brk = np.flatnonzero((np.diff(idx) > 1) | (np.diff(lab[idx]) != 0))
        starts = np.concatenate([[idx[0]], idx[brk + 1]]); ends = np.concatenate([idx[brk], [idx[-1]]]) + 1
        for s, e in zip(starts, ends):
            out.append((int(s), int(e), cats[lab[s]], float(tot[s:e].mean() / len(lists))))
        return out

    def detect(self, text: str, threshold: float | None = None) -> list[Span]:
        orig = text; cmap = None
        if any(ch in _ZW or _is_vs(ch) for ch in text):
            keep = [i for i, ch in enumerate(text) if not (ch in _ZW or _is_vs(ch))]
            text = "".join(orig[i] for i in keep); cmap = keep
        thr = self.threshold if threshold is None else threshold
        ids, offs = tokenize(text)
        if len(ids) == 0: return []
        lps = self.expert_logprobs(ids)
        if self.cfg.mode == "vote" and len(lps) > 1:
            raw = self._vote(len(text), [self._spans(text, lp, offs, thr, "") for lp in lps])
        elif self.cfg.mode == "soft":
            lp = sum(w * l for w, l in zip(self.w, lps)) if len(lps) > 1 else lps[0].copy()
            lp[:, self.tb.o_id] -= self.cfg.bias
            m = lp.max(-1, keepdims=True); z = lp - m; lp = z - np.log(np.exp(z).sum(-1, keepdims=True))
            # score を 1 にする: ルールが同じ範囲を別の分類で出したときは、文脈を読んだモデルの分類を残す
            raw = [(cs, ce, cat, 1.0) for cs, ce, cat, _ in self._spans(text, lp, offs, thr, "")]
        else:
            lp = lps[0] if len(lps) == 1 else sum(w * l for w, l in zip(self.w, lps))
            raw = self._spans(text, lp, offs, thr, "")
        spans = []
        for cs, ce, cat, score in raw:
            if cmap is not None:
                ocs = cmap[cs]; oce = cmap[ce - 1] + 1
                while oce < len(orig) and _is_vs(orig[oce]): oce += 1
                cs, ce = ocs, oce
            spans.append(Span(cs, ce, cat, orig[cs:ce], "MASK", "", score, f"monju:{self.cfg.mode}"))
        spans.sort(key=lambda s: (s.start, s.end))
        return spans


def load_monju(model_dir: str | Path, threads: int = 4, mode: str | None = None, **kw) -> "MonjuDetector":
    """monju.json に書かれた既定の合議方式（多数決）で検出器を作る。"""
    meta = json.load(open(Path(model_dir) / "monju.json", encoding="utf-8"))
    cfg = {**meta.get("default_config", {}), **({"mode": mode} if mode else {}), **kw}
    if "union_labels" in cfg: cfg["union_labels"] = tuple(cfg["union_labels"])
    return MonjuDetector(model_dir, MonjuConfig(threads=threads, **cfg))
