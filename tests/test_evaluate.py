import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toramon.spans import Span
from toramon.evaluate import Doc, score_doc, aggregate, paired_bootstrap


def mk(text, gold, pred, id="d"):
    return Doc(id=id, text=text, gold=gold, pred=pred)


def test_exact_match_and_partial():
    t = "氏名：山田太郎様 TEL 090-1234-5678"
    gold = [Span(3, 7, "private_person", action="MASK"), Span(13, 26, "private_phone", action="MASK")]
    # 氏名は敬称込みで過剰、電話は完全一致
    pred = [Span(3, 8, "private_person"), Span(13, 26, "private_phone")]
    c = score_doc(mk(t, gold, pred))
    r = aggregate([c])
    assert r["micro"]["tp"] == 1 and r["micro"]["fp"] == 1 and r["micro"]["fn"] == 1
    assert r["per_label"]["private_person"]["fn_partial"] == 1
    assert r["full_coverage_rate"] == 1.0          # 文字はすべて覆われている
    assert r["pii_remaining_doc_rate"] == 0.0
    # 「様」1 文字が不要にマスクされた
    assert abs(r["nontarget_char_mask_rate"] - 1 / (len(t) - 4 - 13)) < 1e-9


def test_keep_is_fp_and_review_excluded():
    t = "納期は2026年7月10日。生年月日は1985年3月15日。"
    gold = [Span(3, 13, "private_date", action="KEEP"), Span(20, 30, "private_date", action="MASK")]
    pred = [Span(3, 13, "private_date"), Span(20, 30, "private_date")]
    r = aggregate([score_doc(mk(t, gold, pred))])
    assert r["micro"]["tp"] == 1 and r["micro"]["fp"] == 1
    gold2 = [Span(3, 13, "private_date", action="REVIEW"), Span(20, 30, "private_date", action="MASK")]
    r2 = aggregate([score_doc(mk(t, gold2, pred))])
    assert r2["micro"]["tp"] == 1 and r2["micro"]["fp"] == 0


def test_wrong_label_but_covered_counted_separately():
    t = "会員番号 M-882901"
    gold = [Span(5, 13, "account_number", action="MASK")]
    pred = [Span(5, 13, "secret")]
    r = aggregate([score_doc(mk(t, gold, pred))])
    assert r["micro"]["tp"] == 0 and r["micro"]["fp"] == 1 and r["micro"]["fn"] == 1
    assert r["full_coverage_rate"] == 1.0 and r["full_coverage_same_label_rate"] == 0.0


def test_negatives_and_remaining():
    neg = mk("型番 AB-1234 の価格は 12,000 円です。", [], [Span(3, 10, "account_number")], id="n1")
    neg2 = mk("本日は晴天なり。", [], [], id="n2")
    pos = mk("山田太郎", [Span(0, 4, "private_person")], [Span(0, 2, "private_person")], id="p1")
    r = aggregate([score_doc(neg), score_doc(neg2), score_doc(pos)])
    assert r["negative_doc_fp_rate"] == 0.5
    assert r["pii_remaining_doc_rate"] == 1.0     # 分母は PII あり文書(1)
    assert r["n_docs_with_pii"] == 1 and r["n_docs_negative"] == 2


def test_empty_denominator_is_none():
    r = aggregate([score_doc(mk("本日は晴天なり。", [], []))])
    assert r["pii_remaining_doc_rate"] is None and r["full_coverage_rate"] is None and r["macro_f1"] is None


def test_bootstrap_ci_contains_point():
    docs = [mk("山田太郎", [Span(0, 4, "private_person")], [Span(0, 4, "private_person")], id=f"a{i}") for i in range(10)]
    docs += [mk("山田太郎", [Span(0, 4, "private_person")], [], id=f"b{i}") for i in range(10)]
    counts = [score_doc(d) for d in docs]
    res = paired_bootstrap(counts, None, [d.id for d in docs], lambda a: a["micro"]["f1"], n_boot=200)
    lo, hi = res["ci95"]
    assert lo <= res["point"] <= hi and 0 < lo < hi < 1
