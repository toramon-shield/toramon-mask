"""HF 形式（OpenAIPrivacyFilterForTokenClassification）チェックポイントの推論アダプタ。

T-O / T-S の学習直後の評価、および逆変換した S の検証に使う。
- デコード: 制約付き BIOES Viterbi（sumeshi C++ 実装と同じ構造: 新規開始は閉状態からのみ、I/E は同カテゴリの B/I からのみ、終端は閉状態）。
  オプションで opf の 6 遷移バイアス（viterbi_calibration.json）を加算できる（O 系のみ意味がある）。
- span スコア = 経路上トークン確率の平均（sumeshi と同じ）、threshold で足切り。
- ラベル写像: label_mapping.yaml の model_S / identity を指定。
- 長文: 最大トークン長を超える場合は halo 付きスライディングウィンドウで logits を得る（n_layer × sliding_window = 8×128 = 1024 token の halo）。
- 位置: HF fast tokenizer の offset_mapping（コードポイント）。
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import yaml

from ..spans import Span

ROOT = Path(__file__).resolve().parents[3]


class BIOESTable:
    def __init__(self, id2label: dict[int, str]):
        n = len(id2label)
        self.n = n
        self.tag = ["O"] * n
        self.cat = [-1] * n
        self.cats: list[str] = []
        idx: dict[str, int] = {}
        self.o_id = 0
        for i in range(n):
            lb = id2label[i]
            if lb == "O" or "-" not in lb:
                self.tag[i] = "O"; self.o_id = i if lb == "O" else self.o_id; continue
            t, c = lb.split("-", 1)
            if t not in "BIES":
                continue
            if c not in idx:
                idx[c] = len(self.cats); self.cats.append(c)
            self.tag[i] = t; self.cat[i] = idx[c]
        self.b_of = {c: -1 for c in range(len(self.cats))}
        self.i_of = {c: -1 for c in range(len(self.cats))}
        for i in range(n):
            if self.tag[i] == "B": self.b_of[self.cat[i]] = i
            if self.tag[i] == "I": self.i_of[self.cat[i]] = i
        self.closed = torch.tensor([t in ("O", "E", "S") for t in self.tag])
        self.fresh = torch.tensor([t in ("O", "B", "S") for t in self.tag])


def viterbi_bioes(logprobs: torch.Tensor, tb: BIOESTable) -> list[int]:
    """logprobs: [T, C] (float32, CPU)。制約付き Viterbi で最良経路を返す。"""
    T, C = logprobs.shape
    NEG = -1e30
    prev = torch.full((C,), NEG)
    prev[tb.fresh] = logprobs[0][tb.fresh]
    bp = torch.full((T, C), -1, dtype=torch.long)
    # I/E 遷移の前駆（B-c, I-c）
    pred_b = torch.tensor([tb.b_of[tb.cat[j]] if tb.tag[j] in ("I", "E") else -1 for j in range(C)])
    pred_i = torch.tensor([tb.i_of[tb.cat[j]] if tb.tag[j] in ("I", "E") else -1 for j in range(C)])
    cont = pred_b >= 0
    for t in range(1, T):
        cur = torch.full((C,), NEG)
        closed_scores = torch.where(tb.closed, prev, torch.full_like(prev, NEG))
        best_closed, arg_closed = closed_scores.max(0)
        # fresh (O/B/S)
        cur[tb.fresh] = best_closed + logprobs[t][tb.fresh]
        bp[t][tb.fresh] = arg_closed
        # continuation (I/E)
        if cont.any():
            sb = torch.where(pred_b >= 0, prev[pred_b.clamp(min=0)], torch.full_like(prev, NEG))
            si = torch.where(pred_i >= 0, prev[pred_i.clamp(min=0)], torch.full_like(prev, NEG))
            use_i = si > sb
            best = torch.where(use_i, si, sb)
            arg = torch.where(use_i, pred_i, pred_b)
            cur[cont] = best[cont] + logprobs[t][cont]
            bp[t][cont] = arg[cont]
        prev = cur
    final = torch.where(tb.closed, prev, torch.full_like(prev, NEG))
    j = int(final.argmax())
    path = [0] * T
    for t in range(T - 1, -1, -1):
        path[t] = j
        if t > 0:
            j = int(bp[t][j])
    return path


def spans_from_path(path: list[int], probs: torch.Tensor, tb: BIOESTable) -> list[tuple[int, int, str, float]]:
    out = []
    begin = -1; cat = -1; acc = 0.0
    for t, j in enumerate(path):
        tag = tb.tag[j]
        if tag == "S":
            out.append((t, t, tb.cats[tb.cat[j]], float(probs[t][j]))); begin = -1
        elif tag == "B":
            begin, cat, acc = t, tb.cat[j], float(probs[t][j])
        elif tag == "I" and begin >= 0 and tb.cat[j] == cat:
            acc += float(probs[t][j])
        elif tag == "E" and begin >= 0 and tb.cat[j] == cat:
            acc += float(probs[t][j]); out.append((begin, t, tb.cats[cat], acc / (t - begin + 1))); begin = -1
        else:
            begin = -1
    return out


_ZW_CHARS = set("\u200b\u200c\u200d\u2060\ufeff\u180e")

def _is_variation_selector(ch: str) -> bool:
    o = ord(ch)
    return 0xFE00 <= o <= 0xFE0F or 0xE0100 <= o <= 0xE01EF

def _is_invisible(ch: str) -> bool:
    return ch in _ZW_CHARS or _is_variation_selector(ch)


class HFTokenClassifier:
    def __init__(self, checkpoint: str, device: str = "cuda", label_space: str = "identity", threshold: float = 0.5,
                 max_tokens: int = 4096, dtype=torch.bfloat16, mapping_path: str | None = None, merge_adjacent: bool | None = None, o_bias: float = 0.0, normalize_invisible: bool = True):
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        t0 = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForTokenClassification.from_pretrained(checkpoint, dtype=dtype).to(device).eval()
        self.device = device
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.tb = BIOESTable(self.id2label)
        self.threshold = threshold
        self.normalize_invisible = normalize_invisible
        self.o_bias = float(o_bias)  # O ラベルの対数尤度から引く値（>0 で再現率寄り）。span スコアは元の確率で計算する
        self.max_tokens = max_tokens
        cfg = self.model.config
        self.halo = int(getattr(cfg, "num_hidden_layers", 8)) * int(getattr(cfg, "sliding_window", 128) or 128)
        m = yaml.safe_load(open(mapping_path or ROOT / "configs/label_mapping.yaml", encoding="utf-8"))
        if label_space == "identity":
            self.label_map = None; self.drop_others = False; self.merge_enabled = False
        else:
            ms = m["model_S"]; self.label_map = dict(ms["map"]); self.drop_others = bool(ms.get("drop_others", True))
            mg = ms.get("merge_adjacent_same_category", {}); self.merge_enabled = bool(mg.get("enabled", False)); self.merge_gap = int(mg.get("max_gap_chars", 0))
        if merge_adjacent is not None:
            self.merge_enabled = merge_adjacent
        self.load_seconds = time.perf_counter() - t0
        self.checkpoint = checkpoint

    @torch.inference_mode()
    def token_logprobs(self, text: str) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc["input_ids"]; offs = enc["offset_mapping"]
        n = len(ids)
        if n == 0:
            return torch.zeros((0, self.tb.n)), []
        W = self.max_tokens; halo = self.halo
        out = torch.empty((n, self.tb.n), dtype=torch.float32)
        if n <= W:
            windows = [(0, n, 0, n)]
        else:
            if W <= 2 * halo:
                raise ValueError(f"window {W} must exceed 2*halo={2*halo} for exact windowing")
            stride = W - 2 * halo; windows = []; start = 0
            while True:
                wlen = min(W, n - start); lo = 0 if start == 0 else halo; hi = wlen if start + wlen >= n else wlen - halo
                windows.append((start, wlen, lo, hi))
                if start + wlen >= n: break
                start += stride
        for start, wlen, lo, hi in windows:
            x = torch.tensor([ids[start:start + wlen]], device=self.device)
            logits = self.model(input_ids=x).logits[0].float()
            out[start + lo : start + hi] = torch.log_softmax(logits, -1)[lo:hi].cpu()
        return out, [tuple(o) for o in offs]

    def detect(self, text: str, threshold: float | None = None) -> list[Span]:
        # 不可視文字（ゼロ幅・異体字セレクタ）を除いて推論し、span を元の位置へ写像する（gold は氏名内の不可視文字を含む）
        orig_text = text
        keep = [i for i, ch in enumerate(text) if not _is_invisible(ch)]
        if self.normalize_invisible and len(keep) != len(text):
            text = "".join(orig_text[i] for i in keep)
            cmap = keep
        else:
            cmap = None
        lp, offs = self.token_logprobs(text)
        if lp.shape[0] == 0:
            return []
        lp_dec = lp
        if self.o_bias:
            lp_dec = lp.clone(); lp_dec[:, self.tb.o_id] -= self.o_bias
        path = viterbi_bioes(lp_dec, self.tb)
        probs = lp.exp()
        thr = self.threshold if threshold is None else threshold
        spans: list[Span] = []
        for tb_, te_, cat, score in spans_from_path(path, probs, self.tb):
            if score < thr:
                continue
            label = cat
            if self.label_map is not None:
                label = self.label_map.get(cat)
                if label is None:
                    if self.drop_others: continue
                    label = cat
            cs, ce = offs[tb_][0], offs[te_][1]
            # 先頭空白（o200k は先頭スペースをトークンに含める）と U+3000・'=' のトリム（sumeshi と同条件）
            while cs < ce and (text[cs].isspace() or text[cs] == "="): cs += 1
            while ce > cs and text[ce - 1].isspace(): ce -= 1
            if cs >= ce: continue
            if cmap is not None:
                ocs = cmap[cs]; oce = cmap[ce - 1] + 1
                # 直後の異体字セレクタは文字の一部なので span に含める（ゼロ幅スペース等は含めない）
                while oce < len(orig_text) and _is_variation_selector(orig_text[oce]): oce += 1
                cs, ce = ocs, oce
            spans.append(Span(cs, ce, label, orig_text[cs:ce], "MASK", "", score, f"HF:{cat}"))
        text = orig_text
        spans.sort(key=lambda s: (s.start, s.end))
        if self.merge_enabled:
            merged: list[Span] = []
            for s in spans:
                if merged and merged[-1].label == s.label and 0 <= s.start - merged[-1].end <= self.merge_gap and not text[merged[-1].end:s.start].strip():
                    p = merged[-1]; merged[-1] = Span(p.start, max(p.end, s.end), p.label, text[p.start:max(p.end, s.end)], "MASK", "", min(p.score or 1, s.score or 1), p.source)
                else:
                    merged.append(s)
            spans = merged
        return spans
