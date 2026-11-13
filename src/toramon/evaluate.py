"""共通評価器。

主指標:
  - strict span micro/macro F1（開始・終了・ラベル完全一致、action=MASK の正解に対して）
  - カテゴリ別 precision / recall
  - 全範囲保護率（正解 MASK span の全文字がマスク範囲に含まれる割合。ラベル違いは別集計）
  - PII 残存文書率（MASK 対象を含む文書のうち、必要文字が 1 文字でも残った文書の割合）
  - 非対象文字マスク率（正解で残す文字のうち不要にマスクされた文字の割合）
  - 負例文書誤検知率（MASK 対象がない文書で 1 箇所以上マスクした割合）
  - 対応付き bootstrap（文書 or scenario_family 単位）による 95% 信頼区間

採点規約:
  - 正解 = action=MASK の span。KEEP 領域へのマスクは FP。REVIEW 領域への予測は TP/FP/文字集計から除外。
  - すべてのモデルに同じ規約を使う。
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from .spans import COMMON_LABELS, Span


@dataclass
class Doc:
    id: str
    text: str
    gold: list[Span]
    pred: list[Span]
    group: str = ""  # bootstrap の対応付け単位（scenario_family 等）
    meta: dict = field(default_factory=dict)


def _char_mask(spans: Iterable[Span], n: int, labels: bool = False) -> list:
    m: list = [None] * n if labels else [False] * n
    for s in spans:
        for i in range(s.start, min(s.end, n)):
            m[i] = s.label if labels else True
    return m


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


@dataclass
class DocCounts:
    """1 文書分の集計素片（bootstrap で再標本化できるよう文書単位で保持）。"""
    tp: dict = field(default_factory=lambda: defaultdict(int))
    fp: dict = field(default_factory=lambda: defaultdict(int))
    fn: dict = field(default_factory=lambda: defaultdict(int))
    fn_partial: dict = field(default_factory=lambda: defaultdict(int))  # 重なりはあるが不一致
    gold_spans: dict = field(default_factory=lambda: defaultdict(int))
    fully_covered: dict = field(default_factory=lambda: defaultdict(int))        # ラベル問わず全文字マスク
    fully_covered_same_label: dict = field(default_factory=lambda: defaultdict(int))
    has_mask_gold: bool = False
    pii_remaining: bool = False      # 必要文字が 1 文字でも残った
    keep_chars: int = 0              # 正解で残すべき文字数（REVIEW 除外）
    keep_chars_masked: int = 0       # そのうちマスクされた文字数
    is_negative: bool = False        # MASK 対象なし文書
    negative_fp: bool = False        # 負例で 1 箇所以上マスク


def score_doc(doc: Doc) -> DocCounts:
    c = DocCounts()
    n = len(doc.text)
    gold_mask = [g for g in doc.gold if g.action == "MASK"]
    review = [g for g in doc.gold if g.action == "REVIEW"]
    review_chars = _char_mask(review, n)
    pred = list(doc.pred)
    pred_chars = _char_mask(pred, n)
    pred_label_chars = _char_mask(pred, n, labels=True)

    gold_keys = {g.key() for g in gold_mask}
    # REVIEW 領域と重なる予測は採点から除外
    pred_eff = [p for p in pred if not any(review_chars[i] for i in range(p.start, min(p.end, n)))]
    pred_keys = {p.key() for p in pred_eff}

    for g in gold_mask:
        c.gold_spans[g.label] += 1
        if g.key() in pred_keys:
            c.tp[g.label] += 1
        else:
            c.fn[g.label] += 1
            if any(p.overlaps(g) for p in pred_eff):
                c.fn_partial[g.label] += 1
        covered = all(pred_chars[i] for i in range(g.start, g.end))
        if covered:
            c.fully_covered[g.label] += 1
            if all(pred_label_chars[i] == g.label for i in range(g.start, g.end)):
                c.fully_covered_same_label[g.label] += 1
        else:
            c.pii_remaining = True
    for p in pred_eff:
        if p.key() not in gold_keys:
            c.fp[p.label] += 1

    c.has_mask_gold = bool(gold_mask)
    c.is_negative = not gold_mask
    gold_chars = _char_mask(gold_mask, n)
    for i in range(n):
        if review_chars[i] or gold_chars[i]:
            continue
        c.keep_chars += 1
        if pred_chars[i]:
            c.keep_chars_masked += 1
    if c.is_negative and any(pred_chars[i] and not review_chars[i] for i in range(n)):
        c.negative_fp = True
    return c


def aggregate(counts: Sequence[DocCounts], labels: Sequence[str] = COMMON_LABELS) -> dict:
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int); fnp = defaultdict(int)
    gs = defaultdict(int); cov = defaultdict(int); cov_same = defaultdict(int)
    n_pos = n_rem = n_neg = n_negfp = 0
    keep = keep_m = 0
    for c in counts:
        for k in set(c.tp) | set(c.fp) | set(c.fn):
            tp[k] += c.tp[k]; fp[k] += c.fp[k]; fn[k] += c.fn[k]; fnp[k] += c.fn_partial[k]
        for k in c.gold_spans:
            gs[k] += c.gold_spans[k]; cov[k] += c.fully_covered[k]; cov_same[k] += c.fully_covered_same_label[k]
        if c.has_mask_gold:
            n_pos += 1
            n_rem += int(c.pii_remaining)
        if c.is_negative:
            n_neg += 1
            n_negfp += int(c.negative_fp)
        keep += c.keep_chars; keep_m += c.keep_chars_masked
    per_label = {}
    f1s = []
    for lb in labels:
        p, r, f = prf(tp[lb], fp[lb], fn[lb])
        per_label[lb] = {
            "tp": tp[lb], "fp": fp[lb], "fn": fn[lb], "fn_partial": fnp[lb],
            "precision": p, "recall": r, "f1": f,
            "gold_spans": gs[lb],
            "full_coverage_rate": (cov[lb] / gs[lb]) if gs[lb] else None,
            "full_coverage_same_label_rate": (cov_same[lb] / gs[lb]) if gs[lb] else None,
        }
        if gs[lb] or fp[lb]:
            f1s.append(f)
    extra_labels = sorted(set(tp) | set(fp) | set(fn)) and [k for k in (set(tp) | set(fp) | set(fn)) if k not in labels]
    for lb in extra_labels:
        p, r, f = prf(tp[lb], fp[lb], fn[lb])
        per_label[lb] = {"tp": tp[lb], "fp": fp[lb], "fn": fn[lb], "fn_partial": fnp[lb], "precision": p, "recall": r, "f1": f, "gold_spans": gs[lb], "full_coverage_rate": None, "full_coverage_same_label_rate": None}
    TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    P, R, F = prf(TP, FP, FN)
    G = sum(gs.values())
    return {
        "n_docs": len(counts),
        "n_docs_with_pii": n_pos,
        "n_docs_negative": n_neg,
        "micro": {"tp": TP, "fp": FP, "fn": FN, "precision": P, "recall": R, "f1": F},
        "macro_f1": (sum(f1s) / len(f1s)) if f1s else None,
        "per_label": per_label,
        "full_coverage_rate": (sum(cov.values()) / G) if G else None,
        "full_coverage_same_label_rate": (sum(cov_same.values()) / G) if G else None,
        "pii_remaining_doc_rate": (n_rem / n_pos) if n_pos else None,
        "nontarget_char_mask_rate": (keep_m / keep) if keep else None,
        "negative_doc_fp_rate": (n_negfp / n_neg) if n_neg else None,
    }


def paired_bootstrap(
    docs_a: Sequence[DocCounts], docs_b: Sequence[DocCounts] | None, groups: Sequence[str],
    metric: Callable[[dict], float | None], n_boot: int = 1000, seed: int = 0,
) -> dict:
    """文書（または group）単位の対応付き bootstrap。docs_b が None なら A 単独の CI。

    戻り値: {"point": x, "ci95": [lo, hi]} または差分 {"point_diff": d, "ci95": [lo, hi]}（A - B）。
    """
    rng = random.Random(seed)
    by_group: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g or str(i)].append(i)
    keys = list(by_group)
    samples: list[float] = []
    for _ in range(n_boot):
        chosen = [rng.choice(keys) for _ in keys]
        idx = [i for k in chosen for i in by_group[k]]
        ma = metric(aggregate([docs_a[i] for i in idx]))
        if docs_b is None:
            if ma is not None:
                samples.append(ma)
        else:
            mb = metric(aggregate([docs_b[i] for i in idx]))
            if ma is not None and mb is not None:
                samples.append(ma - mb)
    samples.sort()
    if not samples:
        return {"point": None, "ci95": [None, None], "n_boot": 0}
    lo = samples[int(0.025 * len(samples))]
    hi = samples[min(len(samples) - 1, int(0.975 * len(samples)))]
    pa = metric(aggregate(docs_a))
    if docs_b is None:
        return {"point": pa, "ci95": [lo, hi], "n_boot": len(samples)}
    pb = metric(aggregate(docs_b))
    return {"point_a": pa, "point_b": pb, "point_diff": (pa - pb) if pa is not None and pb is not None else None,
            "ci95": [lo, hi], "n_boot": len(samples)}


def breakdown(docs: Sequence[Doc], counts: Sequence[DocCounts], key: str) -> dict:
    """meta[key] ごとの内訳。"""
    groups: dict[str, list[DocCounts]] = defaultdict(list)
    for d, c in zip(docs, counts):
        groups[str(d.meta.get(key, "unknown"))].append(c)
    return {k: aggregate(v) for k, v in sorted(groups.items())}


# ---------------------------------------------------------------------------
# JSONL 入出力
# ---------------------------------------------------------------------------
def load_gold_jsonl(path: str) -> list[Doc]:
    docs: list[Doc] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            gold = [Span(e["start"], e["end"], e["label"], e.get("text", ""), e.get("action", "MASK"), e.get("entity_id", "")) for e in r.get("entities", [])]
            meta = {k: v for k, v in r.items() if k not in ("text", "entities")}
            docs.append(Doc(id=r["id"], text=r["text"], gold=gold, pred=[], group=r.get("scenario_family", r["id"]), meta=meta))
    return docs


def load_pred_jsonl(path: str) -> dict[str, list[Span]]:
    out: dict[str, list[Span]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out[r["id"]] = [Span(e["start"], e["end"], e["label"], e.get("text", ""), "MASK", e.get("entity_id", ""), e.get("score"), e.get("source", "")) for e in r.get("spans", [])]
    return out


def evaluate_files(gold_path: str, pred_path: str, breakdown_keys: Sequence[str] = ("domain", "generator_model", "length_band", "difficulty", "negative_type")) -> dict:
    docs = load_gold_jsonl(gold_path)
    preds = load_pred_jsonl(pred_path)
    missing = [d.id for d in docs if d.id not in preds]
    for d in docs:
        d.pred = preds.get(d.id, [])
    counts = [score_doc(d) for d in docs]
    res = aggregate(counts)
    res["missing_predictions"] = len(missing)
    res["failed_doc_ids"] = missing[:50]
    res["ci95"] = {
        "micro_f1": paired_bootstrap(counts, None, [d.group for d in docs], lambda a: a["micro"]["f1"]),
        "full_coverage_rate": paired_bootstrap(counts, None, [d.group for d in docs], lambda a: a["full_coverage_rate"]),
        "pii_remaining_doc_rate": paired_bootstrap(counts, None, [d.group for d in docs], lambda a: a["pii_remaining_doc_rate"]),
    }
    res["breakdown"] = {k: breakdown(docs, counts, k) for k in breakdown_keys if any(k in d.meta for d in docs)}
    return res


def compare_files(gold_path: str, pred_a: str, pred_b: str, n_boot: int = 1000) -> dict:
    """A と B の対応付き比較（A - B の差の CI）。"""
    docs = load_gold_jsonl(gold_path)
    pa, pb = load_pred_jsonl(pred_a), load_pred_jsonl(pred_b)
    ca, cb = [], []
    for d in docs:
        d.pred = pa.get(d.id, []); ca.append(score_doc(d))
        d.pred = pb.get(d.id, []); cb.append(score_doc(d))
    groups = [d.group for d in docs]
    return {
        "micro_f1": paired_bootstrap(ca, cb, groups, lambda a: a["micro"]["f1"], n_boot),
        "full_coverage_rate": paired_bootstrap(ca, cb, groups, lambda a: a["full_coverage_rate"], n_boot),
        "pii_remaining_doc_rate": paired_bootstrap(ca, cb, groups, lambda a: a["pii_remaining_doc_rate"], n_boot),
        "nontarget_char_mask_rate": paired_bootstrap(ca, cb, groups, lambda a: a["nontarget_char_mask_rate"], n_boot),
        "negative_doc_fp_rate": paired_bootstrap(ca, cb, groups, lambda a: a["negative_doc_fp_rate"], n_boot),
    }


def format_report(res: dict) -> str:
    lines = []
    m = res["micro"]
    lines.append(f"docs={res['n_docs']} (PIIあり {res['n_docs_with_pii']}, 負例 {res['n_docs_negative']})  missing_pred={res.get('missing_predictions', 0)}")
    lines.append(f"strict micro P/R/F1 = {m['precision']:.3f} / {m['recall']:.3f} / {m['f1']:.3f}   macro F1 = {res['macro_f1'] if res['macro_f1'] is None else round(res['macro_f1'],3)}")
    def fmt(x):
        return "N/A" if x is None else f"{x:.4f}"
    lines.append(f"全範囲保護率 = {fmt(res['full_coverage_rate'])} (同ラベル {fmt(res['full_coverage_same_label_rate'])})  PII残存文書率 = {fmt(res['pii_remaining_doc_rate'])}  非対象文字マスク率 = {fmt(res['nontarget_char_mask_rate'])}  負例文書誤検知率 = {fmt(res['negative_doc_fp_rate'])}")
    if "ci95" in res:
        for k, v in res["ci95"].items():
            lo, hi = v["ci95"]
            lines.append(f"  95%CI {k}: {fmt(v['point'])} [{fmt(lo)}, {fmt(hi)}]")
    lines.append(f"{'label':<18}{'gold':>6}{'tp':>6}{'fp':>6}{'fn':>6}{'fn_part':>8}{'P':>8}{'R':>8}{'F1':>8}{'cover':>8}")
    for lb, v in res["per_label"].items():
        if v["gold_spans"] == 0 and v["fp"] == 0:
            continue
        lines.append(f"{lb:<18}{v['gold_spans']:>6}{v['tp']:>6}{v['fp']:>6}{v['fn']:>6}{v['fn_partial']:>8}{v['precision']:>8.3f}{v['recall']:>8.3f}{v['f1']:>8.3f}{fmt(v['full_coverage_rate']):>8}")
    return "\n".join(lines)
