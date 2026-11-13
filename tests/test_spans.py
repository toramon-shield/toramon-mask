import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pytest
from toramon.spans import Span, byte_span_to_cp, byte_to_cp_table, cp_to_byte_table, redact, validate_spans, merge_overlaps


@pytest.mark.parametrize("text", [
    "山田太郎です",                 # 3 バイト文字
    "a山b田c",                      # 混在
    "👨‍👩‍👧 家族 😀 絵文字",            # ZWJ 連結・サロゲート外
    "ｱｲｳ ＡＢＣ １２３",             # 半角カナ・全角英数字
    "が" + "が" + "ゔ",   # 結合文字（か+濁点）
    "山田山田山田 山田",            # 繰り返し
    "été",               # 結合アクセントと合成済み
    "​山田​太郎",          # ゼロ幅スペース
])
def test_byte_cp_roundtrip(text):
    b = text.encode("utf-8")
    cp2b = cp_to_byte_table(text)
    for cs in range(len(text)):
        for ce in range(cs + 1, len(text) + 1):
            bs, be = cp2b[cs], cp2b[ce]
            assert byte_span_to_cp(text, bs, be) == (cs, ce)
            assert b[bs:be].decode("utf-8") == text[cs:ce]


def test_byte_span_mid_char_rounds_outward():
    text = "山田"  # 6 bytes
    # [1,4) は 山 の途中から 田 の途中 → 両文字を含む
    assert byte_span_to_cp(text, 1, 4) == (0, 2)
    assert byte_span_to_cp(text, 0, 2) == (0, 1)


def test_validate_and_redact():
    text = "連絡先は山田花子です。山田花子さんの番号は090-1111-2222。"
    spans = [Span(4, 8, "private_person", "山田花子", "MASK", "p1"),
             Span(11, 15, "private_person", "山田花子", "MASK", "p1"),
             Span(21, 34, "private_phone", "090-1111-2222", "MASK", "ph1")]
    assert validate_spans(text, spans) == []
    out = redact(text, spans)
    assert out == "連絡先は[PERSON_1]です。[PERSON_1]さんの番号は[PHONE_1]。"
    assert redact(text, spans, style="asterisk") == "連絡先は***です。***さんの番号は***。"
    # KEEP は置換しない
    spans2 = spans + [Span(0, 3, "private_date", "連絡先", "KEEP", "d1")]
    assert redact(text, spans2) == out


def test_validate_detects_problems():
    text = "山田太郎"
    assert validate_spans(text, [Span(0, 2, "private_person", "太郎")])  # text mismatch
    assert validate_spans(text, [Span(0, 9, "private_person")])  # out of range
    assert validate_spans(text, [Span(0, 3, "private_person"), Span(2, 4, "private_person")])  # overlap
    with pytest.raises(ValueError):
        Span(3, 3, "private_person")


def test_merge_overlaps_prefers_longest():
    a = Span(0, 4, "private_person", score=0.9)
    b = Span(2, 10, "private_address", score=0.5)
    assert merge_overlaps([a, b]) == [b]
    c = Span(0, 4, "private_phone", score=0.95)
    assert merge_overlaps([a, c]) == [c]
