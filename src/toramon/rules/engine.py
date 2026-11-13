"""ルールエンジン。

- 検出ルール（regex）: 電話・メール・URL・既知形式の識別番号・認証情報。周囲の語（文脈語）で有効化/抑制できる。
- 境界補正ルール: モデル span の端から敬称・助詞・ラベル語・句読点を外す。
- 衝突解決: ルールとモデルの重なりは merge_overlaps（長い方優先、同長は score）。
- 辞書ルール: 顧客別辞書（別トラック。汎用モデルへ埋め込まない）。
ルール定義は configs/rules_R.yaml（ID, purpose, category, pattern, context_require/suppress, priority, examples）。
処理時間を測り、巨大入力・悪意ある文字列での計算量爆発を避けるため、パターンは線形時間を意識し regex モジュールのタイムアウトを使う。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import regex
import yaml

from ..spans import Span, merge_overlaps

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RULES = ROOT / "configs/rules_R.yaml"


@dataclass
class Rule:
    id: str
    kind: str                 # detect | boundary | suppress
    label: str | None
    pattern: regex.Pattern | None
    priority: int = 50
    context_require: regex.Pattern | None = None   # 直前 40 文字に必要な語
    context_suppress: regex.Pattern | None = None  # 直前/直後にあれば抑制
    action: str = "MASK"
    purpose: str = ""
    enabled: bool = True
    examples_pos: list = field(default_factory=list)
    examples_neg: list = field(default_factory=list)


def load_rules(path: str | Path | None = None) -> list[Rule]:
    cfg = yaml.safe_load(open(path or DEFAULT_RULES, encoding="utf-8"))
    rules = []
    for r in cfg["rules"]:
        rules.append(Rule(
            id=r["id"], kind=r["kind"], label=r.get("label"),
            pattern=regex.compile(r["pattern"], regex.V1) if r.get("pattern") else None,
            priority=int(r.get("priority", 50)),
            context_require=regex.compile(r["context_require"], regex.V1) if r.get("context_require") else None,
            context_suppress=regex.compile(r["context_suppress"], regex.V1) if r.get("context_suppress") else None,
            action=r.get("action", "MASK"), purpose=r.get("purpose", ""), enabled=bool(r.get("enabled", True)),
            examples_pos=r.get("examples_pos", []), examples_neg=r.get("examples_neg", []),
        ))
    return rules


HONORIFICS = regex.compile(r"[\s\u3000]*(?:様|さま|さん|君|くん|ちゃん|氏|殿|先生|御中|各位)$")
LEADING_LABEL_WORDS = regex.compile(r"^(?:氏名|お名前|名前|住所|ご住所|TEL|Tel|tel|電話|電話番号|FAX|Fax|Mail|mail|E-?mail|メール|URL|生年月日|会員番号|口座番号|パスワード|password|pass|pw|token|トークン|ApiKey|API\s*Key|APIキー|api_key|secret|Authorization|Bearer)\s*[:：=]?\s*", regex.I)
DATE_TRAILING_PARTICLES = regex.compile(r"(?:に|から|まで|より|の|へ)$")
ZW = "\u200b\u200c\u200d\u2060\ufeff"
TRAILING_PUNCT = regex.compile(r"[、。,.;:：）)」』\]】>＞\s\u3000" + ZW + r"]+$")
LEADING_PUNCT = regex.compile(r"^[（(「『\[【<＜:：、，,\"'“”‘’\s\u3000" + ZW + r"]+")


class RuleEngine:
    def __init__(self, rules: list[Rule] | None = None, timeout_s: float = 2.0, max_chars: int = 2_000_000):
        self.rules = [r for r in (rules or load_rules()) if r.enabled]
        self.detect_rules = sorted([r for r in self.rules if r.kind == "detect"], key=lambda r: -r.priority)
        self.suppress_rules = [r for r in self.rules if r.kind == "suppress"]
        self.boundary_enabled = any(r.kind == "boundary" for r in self.rules)
        self.timeout_s = timeout_s
        self.max_chars = max_chars
        self.last_timing: dict = {}

    # --- 検出 ---------------------------------------------------------------
    def detect(self, text: str) -> list[Span]:
        if len(text) > self.max_chars:
            raise ValueError(f"input too large for rules ({len(text)} chars > {self.max_chars})")
        t0 = time.perf_counter()
        out: list[Span] = []
        for r in self.detect_rules:
            for m in r.pattern.finditer(text, timeout=self.timeout_s):
                s, e = m.span(r.pattern.groupindex and "v" in r.pattern.groupindex and "v" or 0)
                if s >= e:
                    continue
                before = text[max(0, s - 40) : s]
                after = text[e : e + 20]
                if r.context_require and not r.context_require.search(before, timeout=self.timeout_s):
                    continue
                if r.context_suppress and (r.context_suppress.search(before, timeout=self.timeout_s) or r.context_suppress.search(after, timeout=self.timeout_s)):
                    continue
                out.append(Span(s, e, r.label, text[s:e], r.action, "", 1.0, f"R:{r.id}"))
        self.last_timing["detect_ms"] = (time.perf_counter() - t0) * 1000
        return out

    # --- 抑制（KEEP 判定: 企業代表番号・共用メール等） -------------------------
    def suppress(self, text: str, spans: list[Span]) -> list[Span]:
        kept = []
        for s in spans:
            drop = False
            for r in self.suppress_rules:
                if r.label and r.label != s.label:
                    continue
                if r.pattern and r.pattern.fullmatch(s.text, timeout=self.timeout_s):
                    before = text[max(0, s.start - 40) : s.start]
                    if r.context_require is None or r.context_require.search(before, timeout=self.timeout_s):
                        drop = True; break
            if not drop:
                kept.append(s)
        return kept

    # --- 境界補正 -----------------------------------------------------------
    def fix_boundaries(self, text: str, spans: list[Span]) -> list[Span]:
        out = []
        for s in spans:
            seg = text[s.start : s.end]; st, en = s.start, s.end
            m = LEADING_LABEL_WORDS.search(seg)
            if m and m.end() < len(seg):
                st += m.end(); seg = text[st:en]
            m = LEADING_PUNCT.search(seg)
            if m and m.end() < len(seg):
                st += m.end(); seg = text[st:en]
            m = TRAILING_PUNCT.search(seg)
            if m and m.start() > 0:
                en = st + m.start(); seg = text[st:en]
            if s.label == "private_person":
                m = HONORIFICS.search(seg)
                if m and m.start() > 0:
                    en = st + m.start(); seg = text[st:en]
            if s.label == "private_date":
                m = DATE_TRAILING_PARTICLES.search(seg)
                if m and m.start() > 0:
                    en = st + m.start(); seg = text[st:en]
            if en > st:
                out.append(Span(st, en, s.label, text[st:en], s.action, s.entity_id, s.score, s.source + ("|B" if (st, en) != (s.start, s.end) else "")))
        return out

    # --- 統合 ---------------------------------------------------------------
    def apply(self, text: str, model_spans: list[Span], use_detect: bool = True, use_boundary: bool = True, use_suppress: bool = True) -> list[Span]:
        spans = list(model_spans)
        if use_boundary and self.boundary_enabled:
            spans = self.fix_boundaries(text, spans)
        if use_detect:
            spans = merge_overlaps(spans + self.detect(text))
        if use_suppress:
            spans = self.suppress(text, spans)
        return sorted(spans, key=lambda s: (s.start, s.end))
