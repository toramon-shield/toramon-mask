"""Toramon Mask フィルタ本体: detect(text, policy) / redact(text, policy)。

- 検出と置換を分離。置換はプログラムで行い、非対象文字列は書き換えない。
- 判定元（model / rule / boundary）、信頼度、方針・モデル・ルールのバージョン、処理時間を返す。
- 外向けログに元の PII や対応表を含めない（復元表はローカル保護領域に限定・既定では作らない）。
- 失敗時（モデル未読込・タイムアウト・不正エンコーディング・長文未処理）は status="error" を返し、原文を安全判定しない。
- 外部通信なしで完結する（モデルは事前にローカル配置）。
"""
from __future__ import annotations

import time
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

import yaml

from .spans import Span, merge_overlaps, redact as _redact, validate_spans
from .rules import RuleEngine

ROOT = Path(__file__).resolve().parents[2]
POLICY_VERSION = "annotation_policy v0.1 (2026-11-09)"


@dataclass
class Policy:
    """保護方針の実行時パラメータ。"""
    mask_labels: tuple[str, ...] = ("private_person", "private_address", "private_email", "private_phone", "private_date",
                                    "account_number", "private_url", "secret")
    use_rules: bool = True
    use_boundary_fix: bool = True
    use_keep_suppression: bool = True
    threshold: float | None = None       # S 系のみ
    placeholder_style: Literal["label", "asterisk"] = "label"
    max_chars: int = 200_000             # これを超える入力はエラー（長文は呼び出し側で分割し、全チャンクの処理完了を確認する）
    timeout_s: float = 30.0
    version: str = "policy-0.1"


@dataclass
class FilterResult:
    status: Literal["ok", "error"]
    spans: list[dict]
    redacted_text: str | None
    model_version: str
    rules_version: str
    policy_version: str
    elapsed_ms: float
    error: str | None = None
    counts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ToramonFilter:
    def __init__(self, detector, model_version: str, rules_path: str | None = None, policy: Policy | None = None):
        """detector: .detect(text) -> list[Span] を持つアダプタ（SumeshiPF / OpenAIOPF / 学習済み T-*）。"""
        self.detector = detector
        self.model_version = model_version
        self.policy = policy or Policy()
        self.rules = RuleEngine() if rules_path is None else RuleEngine(__import__("toramon.rules.engine", fromlist=["load_rules"]).load_rules(rules_path))
        self.rules_version = yaml.safe_load(open(rules_path or (ROOT / "configs/rules_R.yaml"), encoding="utf-8")).get("version", "?")

    def _check_input(self, text: str) -> str | None:
        if not isinstance(text, str):
            return "input must be str"
        if "�" in text:
            return "input contains U+FFFD replacement characters (invalid encoding upstream)"
        if len(text) > self.policy.max_chars:
            return f"input too long ({len(text)} > {self.policy.max_chars}); split upstream and verify all chunks"
        return None

    def detect(self, text: str, policy: Policy | None = None) -> FilterResult:
        pol = policy or self.policy
        t0 = time.perf_counter()
        err = self._check_input(text)
        if err:
            return FilterResult("error", [], None, self.model_version, self.rules_version, pol.version, 0.0, err)
        if not text:
            return FilterResult("ok", [], "", self.model_version, self.rules_version, pol.version, 0.0)
        try:
            model_spans = self.detector.detect(text) if pol.threshold is None or not hasattr(self.detector, "threshold") else self.detector.detect(text, threshold=pol.threshold)
            spans = list(model_spans)
            if pol.use_rules or pol.use_boundary_fix or pol.use_keep_suppression:
                spans = self.rules.apply(text, spans, use_detect=pol.use_rules, use_boundary=pol.use_boundary_fix, use_suppress=pol.use_keep_suppression)
            spans = [s for s in spans if s.label in pol.mask_labels]
            spans = merge_overlaps(spans)
            problems = validate_spans(text, spans)
            if problems:
                raise ValueError("internal span validation failed: " + "; ".join(problems[:3]))
        except Exception as e:
            return FilterResult("error", [], None, self.model_version, self.rules_version, pol.version,
                                (time.perf_counter() - t0) * 1000, f"{type(e).__name__}: {e}")
        elapsed = (time.perf_counter() - t0) * 1000
        if elapsed > pol.timeout_s * 1000:
            return FilterResult("error", [], None, self.model_version, self.rules_version, pol.version, elapsed, "timeout")
        counts: dict = {}
        for s in spans:
            counts[s.label] = counts.get(s.label, 0) + 1
        # 外向け結果: span の位置・ラベル・判定元・信頼度。text は呼び出し側が持っているので返さない（ログに PII を残さない）
        out_spans = [{"start": s.start, "end": s.end, "label": s.label, "source": s.source, "score": s.score} for s in spans]
        return FilterResult("ok", out_spans, None, self.model_version, self.rules_version, pol.version, elapsed, counts=counts)

    def redact(self, text: str, policy: Policy | None = None, mapping: dict | None = None) -> FilterResult:
        pol = policy or self.policy
        res = self.detect(text, pol)
        if res.status != "ok":
            return res
        spans = [Span(d["start"], d["end"], d["label"], text[d["start"]:d["end"]], "MASK", f"{d['label']}:{text[d['start']:d['end']]}") for d in res.spans]
        # 同一文書内の同一文字列は同じ番号（同姓同名の別人は統合しない: 文字列が同じ場合に限って同一視する）
        res.redacted_text = _redact(text, spans, style=pol.placeholder_style, keep_entity_ids=True, mapping=mapping)
        return res
