import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toramon.rules import RuleEngine, load_rules
from toramon.spans import Span


def test_rule_examples_self_consistent():
    rules = load_rules()
    for r in rules:
        if r.kind != "detect":
            continue
        for ex in r.examples_pos:
            assert r.pattern.search(ex), (r.id, ex)
        for ex in r.examples_neg:
            m = r.pattern.search(ex)
            # 負例はマッチしないか、文脈条件で落ちること（context_require あり）
            assert m is None or r.context_require is not None, (r.id, ex)


def test_detect_and_suppress():
    e = RuleEngine()
    t = "お問い合わせ 0120-330-118（代表）/ 担当 090-1234-5678 / info@example.jp / t.yamada@example.jp"
    spans = e.suppress(t, e.detect(t))
    texts = sorted(s.text for s in spans)
    assert "090-1234-5678" in texts and "t.yamada@example.jp" in texts
    assert "0120-330-118" not in texts and "info@example.jp" not in texts


def test_boundary_fix():
    e = RuleEngine()
    t = "氏名：山田太郎様 TEL:090-1234-5678。"
    spans = [Span(0, 8, "private_person", t[0:8]), Span(9, 26, "private_phone", t[9:26])]
    fixed = e.fix_boundaries(t, spans)
    assert [s.text for s in fixed] == ["山田太郎", "090-1234-5678"]


def test_mynumber_needs_context():
    e = RuleEngine()
    assert any(s.label == "account_number" for s in e.detect("マイナンバー：1234 5678 9012"))
    assert not any(s.label == "account_number" for s in e.detect("注文番号：1234 5678 9012"))


def test_pathological_input_is_bounded():
    e = RuleEngine(timeout_s=1.0)
    t = ("a@" * 20000) + "0" * 50000
    import time
    t0 = time.perf_counter(); e.detect(t); assert time.perf_counter() - t0 < 5
