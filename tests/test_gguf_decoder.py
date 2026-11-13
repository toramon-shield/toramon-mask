"""GGUF 生ロジット用の numpy 復号器（SumeshiPFDecoded の中核）の自動検査。libpf も GGUF も不要。"""
import math
import numpy as np
from toramon.adapters.sumeshi_pf import _BIOESNP, _viterbi_np, _spans_from_path_np


LABELS = ["O", "B-private_person", "I-private_person", "E-private_person", "S-private_person", "B-private_phone", "I-private_phone", "E-private_phone", "S-private_phone"]


def _lp(rows):
    """各トークンで指定ラベルに高い確率を置いた log-prob 行列を作る。"""
    tb = _BIOESNP(LABELS); out = np.full((len(rows), tb.n), math.log(0.01 / (tb.n - 1)))
    for t, lb in enumerate(rows):
        out[t] = math.log(0.01 / (tb.n - 1)); out[t][LABELS.index(lb)] = math.log(0.99)
    return out, tb


def test_viterbi_follows_clear_bioes_path():
    lp, tb = _lp(["O", "B-private_person", "I-private_person", "E-private_person", "O", "S-private_phone"])
    path = _viterbi_np(lp, tb)
    assert [LABELS[j] for j in path] == ["O", "B-private_person", "I-private_person", "E-private_person", "O", "S-private_phone"]
    spans = _spans_from_path_np(path, np.exp(lp), tb)
    assert [(s[0], s[1], s[2]) for s in spans] == [(1, 3, "private_person"), (5, 5, "private_phone")]
    assert all(abs(s[3] - 0.99) < 1e-6 for s in spans)


def test_viterbi_repairs_invalid_transition():
    # I なしで B の直後に O が来る無効列: 制約付き Viterbi は B→O を許さず、E または S 等の閉じた経路に修復する
    lp, tb = _lp(["B-private_person", "O", "O"])
    path = _viterbi_np(lp, tb); tags = [LABELS[j] for j in path]
    assert tags[0] in ("S-private_person", "O", "B-private_person")
    # 経路は必ず閉じた状態（O/E/S）で終わる
    assert tags[-1].split("-")[0] in ("O", "E", "S")
    # B で始まったなら次は I/E でなければならない
    for a, b in zip(tags, tags[1:]):
        if a.startswith("B-") or a.startswith("I-"):
            assert b.startswith("I-") or b.startswith("E-")


def test_o_bias_shifts_toward_entities():
    lp, tb = _lp(["O", "O"]); lp[1][LABELS.index("S-private_phone")] = math.log(0.45); lp[1][0] = math.log(0.5)
    assert LABELS[_viterbi_np(lp, tb)[1]] == "O"
    lp2 = lp.copy(); lp2[:, tb.o_id] -= 2.0
    assert LABELS[_viterbi_np(lp2, tb)[1]] == "S-private_phone"
