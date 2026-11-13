"""span の共通データ構造と位置変換。

位置の規約: 元の文字列に対する Unicode コードポイント単位、0 始まり、終了位置を含まない。
- sumeshi C API は UTF-8 バイト位置を返す → byte_to_cp_table() で変換する。
- JavaScript の UTF-16 位置とも混同しない（本プロジェクトでは扱わない）。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable, Sequence

COMMON_LABELS: tuple[str, ...] = (
    "private_person",
    "private_address",
    "private_email",
    "private_phone",
    "private_date",
    "account_number",
    "private_url",
    "secret",
)
ACTIONS = ("MASK", "KEEP", "REVIEW")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str
    text: str = ""
    action: str = "MASK"
    entity_id: str = ""
    score: float | None = None
    source: str = ""  # model / rule id / reviewer

    def __post_init__(self) -> None:
        if not (0 <= self.start < self.end):
            raise ValueError(f"invalid span range [{self.start},{self.end})")

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end

    def key(self) -> tuple[int, int, str]:
        return (self.start, self.end, self.label)

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["score"] is None:
            d.pop("score")
        if not d["source"]:
            d.pop("source")
        return d


def validate_spans(text: str, spans: Iterable[Span]) -> list[str]:
    """text[start:end] と span.text の一致、範囲、重なりを検査し、問題を文字列で返す。"""
    problems: list[str] = []
    n = len(text)
    ordered = sorted(spans, key=lambda s: (s.start, s.end))
    for s in ordered:
        if s.end > n:
            problems.append(f"span end {s.end} > len(text) {n}")
            continue
        if s.text and text[s.start : s.end] != s.text:
            problems.append(f"text mismatch at [{s.start},{s.end}): {text[s.start:s.end]!r} != {s.text!r}")
        if s.label not in COMMON_LABELS and not s.label.startswith("ext_"):
            problems.append(f"unknown label {s.label!r}")
        if s.action not in ACTIONS:
            problems.append(f"unknown action {s.action!r}")
    for a, b in zip(ordered, ordered[1:]):
        if a.overlaps(b):
            problems.append(f"overlap [{a.start},{a.end}) {a.label} / [{b.start},{b.end}) {b.label}")
    return problems


# ---------------------------------------------------------------------------
# UTF-8 バイト位置 <-> コードポイント位置
# ---------------------------------------------------------------------------
def byte_to_cp_table(text: str) -> list[int]:
    """UTF-8 バイト位置 b (0..len(bytes)) → コードポイント位置の対応表。

    文字途中のバイト位置（不正な境界）は、その文字の開始コードポイント位置に丸める（
    終了位置として使う場合は cp_end() を使う）。
    """
    table: list[int] = []
    for cp_idx, ch in enumerate(text):
        nbytes = len(ch.encode("utf-8"))
        table.extend([cp_idx] * nbytes)
    table.append(len(text))
    return table


def byte_span_to_cp(text: str, bstart: int, bend: int, table: list[int] | None = None) -> tuple[int, int]:
    """UTF-8 バイト区間 [bstart, bend) をコードポイント区間へ変換する。

    bend が文字途中を指す場合は、その文字を含める（end を切り上げる）。
    """
    table = table or byte_to_cp_table(text)
    if not (0 <= bstart <= bend <= len(table) - 1):
        raise ValueError(f"byte span out of range [{bstart},{bend}) for {len(table)-1} bytes")
    cs = table[bstart]
    ce = table[bend]
    if bend < len(table) - 1 and table[bend] == table[bend - 1] and bend > bstart:
        # bend が文字の途中 → その文字を含める
        ce = table[bend] + 1
    if ce <= cs:
        ce = cs + 1
    return cs, ce


def cp_to_byte_table(text: str) -> list[int]:
    """コードポイント位置 → UTF-8 バイト位置（0..len(text) まで、末尾は総バイト数）。"""
    out = [0]
    acc = 0
    for ch in text:
        acc += len(ch.encode("utf-8"))
        out.append(acc)
    return out


# ---------------------------------------------------------------------------
# 置換（検出とは分離）
# ---------------------------------------------------------------------------
def redact(text: str, spans: Sequence[Span], style: str = "label", keep_entity_ids: bool = True) -> str:
    """spans（重なりなし・MASK のみ）をプレースホルダへ置換する。

    style:
      "label"     -> [PERSON] 等（同一 entity_id は同じ番号: [PERSON_1]）
      "asterisk"  -> ***
    非対象文字列は一切書き換えない。
    """
    placeholder_names = {
        "private_person": "PERSON",
        "private_address": "ADDRESS",
        "private_email": "EMAIL",
        "private_phone": "PHONE",
        "private_date": "DATE",
        "account_number": "ACCOUNT",
        "private_url": "URL",
        "secret": "SECRET",
    }
    masked = [s for s in spans if s.action == "MASK"]
    masked.sort(key=lambda s: s.start)
    pieces: list[str] = []
    cursor = 0
    counters: dict[str, int] = {}
    ids: dict[tuple[str, str], int] = {}
    for s in masked:
        if s.start < cursor:
            raise ValueError("overlapping spans passed to redact()")
        pieces.append(text[cursor : s.start])
        if style == "asterisk":
            pieces.append("***")
        else:
            name = placeholder_names.get(s.label, s.label.upper())
            if keep_entity_ids and s.entity_id:
                k = (s.label, s.entity_id)
                if k not in ids:
                    counters[name] = counters.get(name, 0) + 1
                    ids[k] = counters[name]
                pieces.append(f"[{name}_{ids[k]}]")
            else:
                pieces.append(f"[{name}]")
        cursor = s.end
    pieces.append(text[cursor:])
    return "".join(pieces)


def merge_overlaps(spans: Sequence[Span], prefer: str = "longest") -> list[Span]:
    """重なる span を解決する（ルール・モデル衝突の既定: 長い方を残す。同長なら score 高い方）。"""
    ordered = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
    kept: list[Span] = []
    for s in ordered:
        if kept and s.overlaps(kept[-1]):
            a = kept[-1]
            la, ls = a.end - a.start, s.end - s.start
            if ls > la or (ls == la and (s.score or 0) > (a.score or 0)):
                kept[-1] = s
            continue
        kept.append(s)
    return kept
