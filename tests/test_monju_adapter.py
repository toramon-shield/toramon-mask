"""Monju アダプタのうち、重みなしで確かめられる部分（BIOES の復号、トークンの文字位置、投票）。"""
import numpy as np
import pytest

mo = pytest.importorskip("toramon.adapters.monju_onnx")


def _labels():
    cats = ["account_number", "private_address", "private_date", "private_email", "private_person", "private_phone", "private_url", "secret"]
    return ["O"] + [f"{p}-{c}" for c in cats for p in "BIES"]


def test_decode_recovers_a_single_span():
    labels = _labels(); tb = mo.BIOES(labels); n = 6
    lp = np.full((n, len(labels)), -20.0, dtype=np.float32); lp[:, 0] = -0.01
    for t, tag in ((2, "B-private_person"), (3, "E-private_person")):
        lp[t, 0] = -20.0; lp[t, labels.index(tag)] = -0.01
    spans = mo.decode(lp, tb)
    assert [(s[0], s[1], s[2]) for s in spans] == [(2, 3, "private_person")]


def test_tokenize_offsets_cover_text():
    pytest.importorskip("tiktoken")
    text = "山田太郎様の電話は090-1234-5678です。"
    ids, offs = mo.tokenize(text)
    assert len(ids) == len(offs) and offs[0][0] == 0 and offs[-1][1] == len(text)
    assert all(offs[i][0] <= offs[i + 1][0] for i in range(len(offs) - 1))


def test_vote_quorum_person_needs_one_vote_phone_needs_two():
    det = mo.MonjuDetector.__new__(mo.MonjuDetector)
    det.tb = mo.BIOES(_labels()); det.cfg = mo.MonjuConfig(mode="vote", k=2, union_labels=("private_person", "private_address"))
    a = [(0, 4, "private_person", 0.9)]; b = [(9, 22, "private_phone", 0.9)]; c = []
    out = det._vote(25, [a, b, c])
    assert [(s, e, lab) for s, e, lab, _ in out] == [(0, 4, "private_person")]
    out = det._vote(25, [a, b, b])
    assert [(s, e, lab) for s, e, lab, _ in out] == [(0, 4, "private_person"), (9, 22, "private_phone")]
