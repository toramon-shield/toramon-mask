"""Toramon Gateway の純粋関数（プレースホルダ割当・復元、本文走査）の検査。モデルは使わない。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import toramon_gateway as g


class _FakeFilt:
    model_version = "fake"; rules_version = "0"
    def detect(self, text):
        import re
        spans = [{"start": m.start(), "end": m.end(), "label": "private_phone"} for m in re.finditer(r"0\d0-\d{4}-\d{4}", text)]
        spans += [{"start": m.start(), "end": m.end(), "label": "private_person"} for m in re.finditer(r"山田太郎", text)]
        class R: status = "ok"; elapsed_ms = 1.0; error = None
        r = R(); r.spans = spans; return r


def test_mask_and_restore_roundtrip():
    m = g.Masker(_FakeFilt())
    s = "山田太郎 090-1234-5678 と 山田太郎 080-0000-1111"
    red = m.mask(s)
    assert red == "[PERSON_1] [PHONE_1] と [PERSON_1] [PHONE_2]"
    assert m.counts["private_person"] == 2 and m.counts["private_phone"] == 2
    assert m.restore("了解: [PERSON_1] へ [PHONE_2] で連絡") == "了解: 山田太郎 へ 080-0000-1111 で連絡"


def test_walk_text_covers_chat_and_responses_shapes():
    m = g.Masker(_FakeFilt())
    req = {"model": "x", "messages": [{"role": "user", "content": "山田太郎です"}, {"role": "user", "content": [{"type": "text", "text": "090-1234-5678"}]}],
           "input": [{"role": "user", "content": [{"type": "input_text", "text": "山田太郎"}]}]}
    g.walk_text(req, m.mask)
    assert req["messages"][0]["content"] == "[PERSON_1]です"
    assert req["messages"][1]["content"][0]["text"] == "[PHONE_1]"
    assert req["input"][0]["content"][0]["text"] == "[PERSON_1]"
