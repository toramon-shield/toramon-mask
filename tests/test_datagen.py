import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pytest

from toramon.spans import validate_spans
from toramon.datagen import (
    KIND_SPECS,
    FillError,
    Pool,
    fill_document,
    is_valid_corp_number,
    is_valid_my_number,
    my_number_check_digit,
    corp_number_check_digit,
    to_skeleton,
)
from toramon.datagen import _data_names as N
from toramon.datagen import _data_places as P

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "configs" / "data_schema.json").read_text(encoding="utf-8"))
ENTITY_PROPS = set(SCHEMA["properties"]["entities"]["items"]["properties"])
ENTITY_REQUIRED = set(SCHEMA["properties"]["entities"]["items"]["required"])
LABELS = set(SCHEMA["properties"]["entities"]["items"]["properties"]["label"]["enum"])


def _fill(template, split="train", seed=1, placeholders=None):
    pool = Pool(split, seed=seed)
    return fill_document(template, placeholders, pool, random.Random(seed))


def _check_doc(doc):
    assert validate_spans(doc.text, doc.spans()) == []
    for e in doc.entities:
        assert doc.text[e["start"] : e["end"]] == e["text"]
        assert ENTITY_REQUIRED <= set(e) <= ENTITY_PROPS
        assert e["label"] in LABELS
        assert e["source"] == "placeholder"
    for k in doc.keep_values:
        assert doc.text[k["start"] : k["end"]] == k["value"]
    assert "{{" not in doc.text and "}}" not in doc.text


# ---------------------------------------------------------------------------
# 基礎データ量
# ---------------------------------------------------------------------------
def test_data_volume():
    assert len(N.SURNAMES_COMMON) + len(N.SURNAMES_RARE) >= 250
    assert len(N.GIVEN_COMMON) + len(N.GIVEN_RARE) >= 250
    assert len(N.FOREIGN_GIVEN) + len(N.FOREIGN_SURNAMES) >= 60
    genders = {g[3] for g in N.GIVEN_COMMON}
    assert {"m", "f"} <= genders
    assert len(P.PREFECTURES) == 47
    assert len(P.MUNICIPALITIES) >= 150
    assert len(P.TOWNS) >= 50
    # 読みはひらがな、ローマ字は ASCII
    for k, kana, ro in N.SURNAMES_COMMON + N.SURNAMES_RARE:
        assert re.fullmatch(r"[ぁ-ゖー]+", kana), (k, kana)
        assert ro.isascii() and ro[0].isupper(), (k, ro)
    for k, kana, ro, g in N.GIVEN_COMMON + N.GIVEN_RARE:
        assert re.fullmatch(r"[ぁ-ゖー]+", kana), (k, kana)
        assert ro.isascii() and g in "mfu"
    # 旧字体・異体字が希少姓に含まれる
    joined = "".join(s[0] for s in N.SURNAMES_RARE)
    for ch in "髙﨑邊齋":
        assert ch in joined


def test_email_domains_are_reserved():
    for d in P.EMAIL_DOMAINS + P.COMPANY_DOMAINS:
        assert re.search(r"(^|\.)example\.(jp|com|net|org|test)$|\.example$|\.invalid$|\.test$", d), d


# ---------------------------------------------------------------------------
# チェックデジット
# ---------------------------------------------------------------------------
def test_my_number_check_digit_hand_computed():
    # 右から P_n×Q_n: 1*2+0*3+9*4+8*5+7*6+6*7 + 5*2+4*3+3*4+2*5+1*6 = 212, 212 mod 11 = 3, 11-3 = 8
    assert my_number_check_digit("12345678901") == 8
    assert is_valid_my_number("123456789018")
    assert not is_valid_my_number("123456789017")
    assert is_valid_my_number("1234 5678 9018")
    # mod が 1 以下なら 0: 全桁 0 → 0
    assert my_number_check_digit("00000000000") == 0
    assert is_valid_my_number("000000000000")


def test_my_number_generation_valid_and_invalid_flagged():
    pool = Pool("train", seed=5)
    rng = random.Random(5)
    for _ in range(100):
        v = pool.my_number(rng, valid=True)
        assert v.meta["checksum_valid"] is True and is_valid_my_number(v.value) and len(v.value) == 12
        w = pool.my_number(rng, valid=False)
        assert w.meta["checksum_valid"] is False and not is_valid_my_number(w.value) and len(w.value) == 12


def test_corp_number_check_digit():
    assert corp_number_check_digit("000000000001") == 8
    pool = Pool("dev", seed=1)
    for _ in range(50):
        v = pool.corp_number()
        assert len(v.value) == 13 and is_valid_corp_number(v.value)


# ---------------------------------------------------------------------------
# 位置の正確さ
# ---------------------------------------------------------------------------
TEMPLATES = [
    "氏名：{{PERSON_1}}様（{{PERSON_KANA_1}}）\nTEL：{{PHONE_1:fullwidth}}／FAX：{{FAX_1}}",
    "😀{{PERSON_1:furigana}}😀 👨‍👩‍👧 {{PERSON_1:kyujitai}} 🎉{{EMAIL_1}}🎉",
    "お届け先:\n{{ADDRESS_1:multiline}}\n{{PERSON_1:fullspace}}　様",
    "{{ADDRESS_1:fullwidth}}／{{POSTAL_1:fullwidth}}／{{ADDRESS_PARTIAL_1}}／{{ADDRESS_1:kanji_style}}",
    "生年月日: {{DATE_BIRTH_1:wareki}} / {{DATE_BIRTH_1:wareki_short}} / {{DATE_BIRTH_1:fullwidth}} / {{DATE_BIRTH_1:weekday}}",
    "{{PERSON_1:joint}}、{{PERSON_2:joint_space}}、{{PERSON_3:romaji_upper}}、{{PERSON_ROMAJI_3:initial}}",
    "ID: {{MYNUMBER_1:invalid+spaced4}} / {{MYNUMBER_2:fullwidth}} / {{LICENSE_1}} / {{PASSPORT_1}} / {{PENSION_1}} / {{INSURANCE_1:labeled}} / {{ACCOUNT_1:labeled}} / {{ACCOUNT_2}} / {{RESIDENCE_CARD_1}}",
    "{{MEMBER_ID_1}} {{CUSTOMER_ID_1}} {{CONTRACT_ID_1}} {{ORDER_ID_1}} {{EMPLOYEE_ID_1}} {{CORP_NUMBER_1}}",
    "URL: {{URL_TRACK_1}}。{{URL_PERSONAL_1:noscheme}}）{{PUBLIC_URL_1}}",
    'password="{{SECRET_PASSWORD_1}}" API_KEY={{SECRET_APIKEY_1}} Authorization: {{SECRET_TOKEN_1:bearer}} code={{SECRET_OTP_1}} DSN={{SECRET_CONNSTR_1:kv}}',
    "{{COMPANY_1}} {{COMPANY_ADDRESS_1}} 代表 {{COMPANY_PHONE_1:paren}} {{SHARED_EMAIL_1}} 納期 {{DATE_BUSINESS_1:slash}} {{PRODUCT_CODE_1}} {{VERSION_1}} {{ERROR_CODE_1}} {{AMOUNT_1}} {{INVOICE_NO_1}} {{PLACE_1}} {{PUBLIC_PERSON_1}}",
    "{{PERSON_1:foreign}} / {{PERSON_KANA_1}} / {{PERSON_FAMILY_1}} / {{PERSON_1:handle}} / {{EMAIL_1:fullwidth_at}} / {{URL_PERSONAL_1}}",
    "{{PHONE_1}} {{PHONE_1:paren2}} {{PHONE_1:nosep}} {{PHONE_1:intl_space}} {{PHONE_1:ext}} {{PHONE_INTL_1:intl_paren0}} {{PHONE_2:mobile}}",
    "{{PERSON_1:space}}{{PERSON_2}}{{PERSON_3:hiragana}}",  # 隣接
    "{{PERSON}}だけ（添字なし）",
]


@pytest.mark.parametrize("template", TEMPLATES)
@pytest.mark.parametrize("split", ["train", "dev", "test"])
def test_offsets_all_variants(template, split):
    for seed in range(4):
        doc = _fill(template, split, seed)
        _check_doc(doc)
        assert len(doc.fills) == len(re.findall(r"\{\{", template))


def test_fullwidth_and_linebreak_values():
    doc = _fill("TEL:{{PHONE_1:fullwidth}} 〒{{POSTAL_1:nomark}}\n{{ADDRESS_1:multiline}}", "train", 2)
    _check_doc(doc)
    phone = doc.entities[0]["text"]
    assert re.fullmatch(r"[０-９－]+", phone)
    addr = [e for e in doc.entities if e["entity_id"] == "address-001" and "\n" in e["text"]]
    assert addr, "multiline address must contain line breaks inside the span"


def test_emoji_and_astral_offsets():
    # 𠮷 (U+20BB7) は BMP 外。前後に絵文字。
    pool = Pool("train", seed=0)
    pool._common_surnames = (("𠮷田", "よしだ", "Yoshida"),)
    pool._rare_surnames = ()
    doc = fill_document("🎉{{PERSON_1}}🎉{{PERSON_FAMILY_1:kyujitai}}👨‍👩‍👧{{PERSON_KANA_1}}", None, pool, random.Random(0))
    _check_doc(doc)
    assert doc.entities[0]["start"] == 1 and doc.entities[0]["text"].startswith("𠮷田")


def test_kyujitai_variant_changes_glyph():
    doc = _fill("{{PERSON_1:kyujitai}} / {{PERSON_1}}", "test", 3)
    kyu, shin = doc.entities[0]["text"], doc.entities[1]["text"]
    assert kyu != shin and doc.entities[0]["subtype"] == "kyujitai"


# ---------------------------------------------------------------------------
# entity_id の一貫性と派生
# ---------------------------------------------------------------------------
def test_entity_id_consistency():
    doc = _fill("{{PERSON_1}} {{PERSON_KANA_1}} {{PERSON_FAMILY_1}} {{PERSON_ROMAJI_1}} {{PERSON_2}} {{PERSON_FAMILY_2}} "
                "{{ADDRESS_1}} {{POSTAL_1}} {{ADDRESS_PARTIAL_1}} {{PHONE_1}} {{PHONE_INTL_1}} {{FAX_1}} {{EMAIL_1}} {{EMAIL_2}}", "train", 11)
    ids = {e["text"]: e["entity_id"] for e in doc.entities}
    by_id = {}
    for e in doc.entities:
        by_id.setdefault(e["entity_id"], []).append(e)
    assert len(by_id["person-001"]) == 4 and len(by_id["person-002"]) == 2
    assert len(by_id["address-001"]) == 3 and len(by_id["phone-001"]) == 2 and len(by_id["fax-001"]) == 1
    assert {e["subtype"] for e in by_id["person-001"] if "subtype" in e} == {"furigana", "family_name", "romaji"}
    # PERSON_FAMILY_1 は PERSON_1 の姓
    p1 = doc.values["person-001"]
    fam = next(e for e in by_id["person-001"] if e.get("subtype") == "family_name")
    assert fam["text"] == p1.surname
    assert doc.values["person-002"].surname + doc.values["person-002"].given != p1.surname + p1.given
    # EMAIL_1 は PERSON_1 のローマ字から派生、EMAIL_2 は PERSON_2 から
    g1, s1 = p1.ascii_tokens()
    email1 = by_id["email-001"][0]["text"].split("@")[0]
    assert s1 in email1 or g1 in email1
    g2, s2 = doc.values["person-002"].ascii_tokens()
    email2 = by_id["email-002"][0]["text"].split("@")[0]
    assert s2 in email2 or g2 in email2
    # PHONE_INTL_1 は PHONE_1 と同じ番号
    ph = doc.values["phone-001"]
    intl = next(e for e in by_id["phone-001"] if e.get("subtype") == "intl")
    assert intl["text"].startswith("+81") and ph.area[1:] in intl["text"]


def test_joint_and_furigana_parts():
    doc = _fill("{{PERSON_1:joint}}／{{PERSON_1:furigana}}", "train", 4)
    _check_doc(doc)
    joint = [e for e in doc.entities if e["start"] < doc.text.index("／")]
    assert len(joint) == 2
    assert joint[0]["entity_id"] == "person-001" and joint[1]["entity_id"] == "person-001-joint"
    assert joint[1]["subtype"] == "joint_name"
    assert doc.text[joint[0]["end"]] == "・"  # 区切り文字は span 外
    furi = [e for e in doc.entities if e["start"] > doc.text.index("／")]
    assert len(furi) == 2 and furi[1]["subtype"] == "furigana" and furi[1]["entity_id"] == "person-001"
    assert doc.text[furi[0]["end"]] == "（" and doc.text[furi[1]["end"]] == "）"


def test_email_derivation_when_person_appears_later():
    doc = _fill("{{EMAIL_1}} ... {{PERSON_1}}", "train", 8)
    g, s = doc.values["person-001"].ascii_tokens()
    local = doc.entities[0]["text"].split("@")[0]
    assert s in local or g in local


# ---------------------------------------------------------------------------
# label / action の対応
# ---------------------------------------------------------------------------
def test_keep_kinds_labels_and_keep_values():
    doc = _fill("{{COMPANY_PHONE_1}} {{SHARED_EMAIL_1}} {{DATE_BUSINESS_1}} {{PUBLIC_URL_1}} {{COMPANY_ADDRESS_1}} {{CORP_NUMBER_1}} {{PUBLIC_PERSON_1}} "
                "{{COMPANY_1}} {{PRODUCT_CODE_1}} {{VERSION_1}} {{ERROR_CODE_1}} {{AMOUNT_1}} {{INVOICE_NO_1}} {{PLACE_1}}", "dev", 2)
    _check_doc(doc)
    got = {(e["label"], e["action"]) for e in doc.entities}
    assert got == {("private_phone", "KEEP"), ("private_email", "KEEP"), ("private_date", "KEEP"), ("private_url", "KEEP"),
                   ("private_address", "KEEP"), ("account_number", "KEEP"), ("private_person", "KEEP")}
    assert [k["kind"] for k in doc.keep_values] == ["COMPANY", "PRODUCT_CODE", "VERSION", "ERROR_CODE", "AMOUNT", "INVOICE_NO", "PLACE"]
    cp = next(e for e in doc.entities if e["label"] == "private_phone")
    assert cp["text"].startswith(("0120", "0570", "0800"))
    se = next(e for e in doc.entities if e["label"] == "private_email")
    assert se["text"].split("@")[0] in {"info", "support", "sales", "contact", "inquiry", "cs", "help", "noreply", "no-reply", "admin", "recruit", "press", "order", "billing", "webmaster"}


def test_mask_kinds_labels():
    tpl = " ".join("{{%s_1}}" % k for k, s in KIND_SPECS.items() if s.action == "MASK")
    doc = _fill(tpl, "train", 9)
    _check_doc(doc)
    assert all(e["action"] == "MASK" for e in doc.entities)
    labels = {e["label"] for e in doc.entities}
    assert labels == {"private_person", "private_address", "private_email", "private_phone", "private_date", "account_number", "private_url", "secret"}


def test_placeholders_override_value_action_note():
    ph = {"PERSON_1": {"action": "REVIEW", "note": "ニックネームか判断不能"}, "PRODUCT_CODE_1": {"value": "XR-2200"}, "GADGET_1": {"value": "ガジェットX", "label": "ext_other", "action": "KEEP"}}
    doc = _fill("{{PERSON_1}} {{PRODUCT_CODE_1}} {{GADGET_1}}", "train", 1, ph)
    _check_doc(doc)
    assert doc.entities[0]["action"] == "REVIEW" and doc.entities[0]["note"] == "ニックネームか判断不能"
    assert doc.keep_values[0]["value"] == "XR-2200"
    assert doc.entities[1]["label"] == "ext_other" and doc.entities[1]["text"] == "ガジェットX"


# ---------------------------------------------------------------------------
# エラー系
# ---------------------------------------------------------------------------
def test_unfilled_placeholder_raises():
    with pytest.raises(FillError):
        _fill("{{person_1}} は小文字なので未解決")
    with pytest.raises(FillError):
        _fill("{{UNKNOWN_KIND_1}}")
    with pytest.raises(FillError):
        _fill("{{PERSON_1:no_such_variant}}")


# ---------------------------------------------------------------------------
# 分割の分離
# ---------------------------------------------------------------------------
def _sample(split, n=1500, seeds=(0, 1)):
    persons, addrs, emails, ids = set(), set(), set(), set()
    for seed in seeds:
        pool = Pool(split, seed=seed)
        rng = random.Random(seed * 7919 + 1)
        for _ in range(n):
            p = pool.person(rng)
            persons.add((p.surname, p.given))
            # 漢字以外の描画（かな・ローマ字）も split 間で重ならないこと（斎藤/齋藤→Saito 等の同音異字）
            for v in ("romaji", "katakana", "hiragana", "romaji_upper"):
                persons.add(p.render("PERSON", v).text)
            a = pool.address(rng)
            addrs.add((a.municipality, a.town))
            emails.add(pool.email(rng, person=p).local)
            ids.add(pool.my_number(rng).value)
            ids.add(pool.member_id(rng).value)
            ids.add(pool.order_id(rng).value)
    return persons, addrs, emails, ids


def test_pool_disjoint_across_splits():
    s = {sp: _sample(sp) for sp in ("train", "dev", "test")}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        for i, what in enumerate(("person combos", "address combos", "email locals", "ids")):
            assert not (s[a][i] & s[b][i]), f"{what} leak between {a} and {b}: {list(s[a][i] & s[b][i])[:5]}"
    # それぞれ十分な多様性がある
    assert len(s["train"][0]) > 1000 and len(s["test"][0]) > 500 and len(s["dev"][0]) > 300


def test_rare_surnames_partitioned_and_test_has_exclusive_share():
    pools = {sp: Pool(sp, seed=0) for sp in ("train", "dev", "test")}
    rare = {sp: {s[0] for s in pools[sp]._rare_surnames} for sp in pools}
    assert not (rare["train"] & rare["test"]) and not (rare["dev"] & rare["test"]) and not (rare["train"] & rare["dev"])
    assert len(rare["test"]) >= 10
    # 全 split を合わせると全希少姓を覆う
    assert set().union(*rare.values()) == {s[0] for s in N.SURNAMES_RARE}
    # train から引いた人物に test 専用の希少姓は現れない
    rng = random.Random(3)
    seen = {pools["train"].person(rng).surname for _ in range(3000)}
    assert not (seen & rare["test"])


def test_split_assignment_independent_of_seed():
    a = Pool("test", seed=1).describe()
    b = Pool("test", seed=99).describe()
    assert a["PERSON"] == b["PERSON"] and a["ADDRESS"] == b["ADDRESS"]
    assert a["_detail"]["surnames_rare_in_split"] == b["_detail"]["surnames_rare_in_split"]


def test_describe_counts():
    for sp in ("train", "dev", "test"):
        d = Pool(sp, seed=0).describe()
        for kind in KIND_SPECS:
            assert isinstance(d[kind], int) and d[kind] > 0, kind
        assert d["_detail"]["prefectures"] == 47
    assert Pool("train").describe()["PERSON"] > Pool("test").describe()["PERSON"] > Pool("dev").describe()["PERSON"]


def test_determinism():
    tpl = "{{PERSON_1}} {{ADDRESS_1}} {{PHONE_1}} {{EMAIL_1}} {{MYNUMBER_1}} {{SECRET_APIKEY_1}} {{DATE_BIRTH_1:wareki}}"
    assert _fill(tpl, "train", 42).text == _fill(tpl, "train", 42).text
    assert _fill(tpl, "train", 42).text != _fill(tpl, "train", 43).text


# ---------------------------------------------------------------------------
# 値の形式
# ---------------------------------------------------------------------------
def test_phone_formats():
    pool = Pool("train", seed=1)
    rng = random.Random(1)
    for _ in range(200):
        m = pool.phone(rng, kind="mobile")
        assert re.fullmatch(r"0[789]0-\d{4}-\d{4}", m.render("PHONE").text)
        assert re.fullmatch(r"\+81-[789]0-\d{4}-\d{4}", m.render("PHONE_INTL").text)
        assert re.fullmatch(r"\d{11}", m.render("PHONE", "nosep").text)
        l = pool.phone(rng, kind="landline")
        assert re.fullmatch(r"0\d{1,3}-\d{2,4}-\d{4}", l.render("PHONE").text)
        assert len(l.render("PHONE", "nosep").text) == 10
        assert re.fullmatch(r"\(0\d{1,3}\)\d{2,4}-\d{4}", l.render("PHONE", "paren").text)
        c = pool.company_phone(rng)
        assert re.fullmatch(r"(0120-\d{3}-\d{3}|0570-\d{3}-\d{3}|0800-\d{3}-\d{4})", c.render("COMPANY_PHONE").text)


def test_address_formats():
    pool = Pool("test", seed=2)
    rng = random.Random(2)
    munis = {m for _, m in P.MUNICIPALITIES}
    for _ in range(100):
        a = pool.address(rng)
        full = a.render("ADDRESS").text
        assert re.match(r"〒\d{3}-\d{4} ", full)
        assert a.prefecture in P.PREFECTURES and a.municipality in munis and a.town in P.TOWNS
        assert full.startswith(f"〒{a.postal} {a.prefecture}{a.municipality}{a.town}{a.chome}-{a.ban}")
        assert a.render("ADDRESS_PARTIAL").text == a.prefecture + a.municipality
        assert a.render("POSTAL").text == f"〒{a.postal}"
        assert "丁目" in a.render("ADDRESS", "kanji_style").text and "番" in a.render("ADDRESS", "kanji_style").text
        assert not re.search(r"[0-9]", a.render("ADDRESS", "fullwidth").text)
        assert a.render("ADDRESS", "building").text.endswith(a.building + (a.room or ""))
        assert "\n" in a.render("ADDRESS", "multiline").text


def test_date_formats():
    import datetime as dt
    from toramon.datagen import DateVal
    d = DateVal(dt.date(1983, 4, 12), "birth")
    assert d.render("DATE_BIRTH", "wareki").text == "昭和58年4月12日"
    assert d.render("DATE_BIRTH", "wareki_short").text == "S58.4.12"
    assert d.render("DATE_BIRTH", "kanji").text == "1983年4月12日"
    assert d.render("DATE_BIRTH", "slash").text == "1983/04/12"
    assert d.render("DATE_BIRTH", "iso").text == "1983-04-12"
    assert d.render("DATE_BIRTH", "dot_short").text == "83.4.12"
    assert d.render("DATE_BIRTH", "weekday").text == "1983年4月12日（火）"
    assert d.render("DATE_BIRTH", "fullwidth").text == "１９８３年４月１２日"
    r = DateVal(dt.date(2019, 5, 1))
    assert r.render("DATE_PERSONAL", "wareki").text == "令和元年5月1日"
    assert DateVal(dt.date(2026, 7, 3)).render("DATE_BUSINESS", "wareki_year").text == "令和8年"
    assert DateVal(dt.date(1989, 1, 7)).render("DATE_BIRTH", "wareki").text == "昭和64年1月7日"
    assert DateVal(dt.date(1989, 1, 8)).render("DATE_BIRTH", "wareki").text == "平成元年1月8日"


def test_person_name_variants():
    from toramon.datagen import Person
    p = Person("山田", "太郎", "やまだ", "たろう", "Yamada", "Taro", "m", "ja")
    p.joint_given = ("花子", "はなこ", "Hanako")
    R = lambda k, v=None: p.render(k, v).text  # noqa: E731
    assert R("PERSON") == "山田太郎" and R("PERSON", "space") == "山田 太郎" and R("PERSON", "fullspace") == "山田　太郎"
    assert R("PERSON", "hiragana") == "やまだたろう" and R("PERSON", "katakana") == "ヤマダタロウ"
    assert R("PERSON", "romaji") == "Taro Yamada" and R("PERSON", "romaji_upper") == "YAMADA Taro" and R("PERSON", "romaji_sg") == "Yamada Taro"
    assert R("PERSON", "furigana") == "山田太郎（ヤマダタロウ）" and R("PERSON", "furigana_hira") == "山田太郎（やまだたろう）"
    assert R("PERSON", "joint") == "山田太郎・花子"
    assert R("PERSON_KANA") == "ヤマダタロウ" and R("PERSON_KANA", "hiragana") == "やまだたろう" and R("PERSON_KANA", "space") == "ヤマダ タロウ"
    assert R("PERSON_ROMAJI") == "Taro Yamada" and R("PERSON_ROMAJI", "upper") == "YAMADA Taro" and R("PERSON_ROMAJI", "initial") == "T. Yamada"
    assert R("PERSON_FAMILY") == "山田" and R("PERSON_FAMILY", "katakana") == "ヤマダ" and R("PERSON_FAMILY", "romaji") == "Yamada"
    f = Person("スミス", "マイケル", "スミス", "マイケル", "Smith", "Michael", "m", "foreign_kana")
    assert f.render("PERSON").text == "マイケル・スミス" and f.render("PERSON", "romaji_upper").text == "SMITH Michael"
    assert f.render("PERSON_FAMILY").text == "スミス"


def test_email_and_url_hosts_reserved():
    pool = Pool("train", seed=4)
    rng = random.Random(4)
    host_re = re.compile(r"^(https?://)?([^/?#]+)")
    for _ in range(100):
        e = pool.email(rng)
        assert e.domain in P.EMAIL_DOMAINS and re.fullmatch(r"[a-z0-9._-]+", e.local)
        for u in (pool.url_track(rng), pool.url_personal(rng), pool.public_url(rng)):
            host = host_re.match(u.text).group(2)
            assert re.search(r"(^|\.)example\.(jp|com|net|org|test)$|\.example$|\.invalid$|\.test$", host), u.text


def test_id_formats():
    pool = Pool("train", seed=6)
    rng = random.Random(6)
    for _ in range(50):
        assert re.fullmatch(r"\d{12}", pool.license_number(rng).value)
        assert re.fullmatch(r"[A-Z]{2}\d{7}", pool.passport_number(rng).value)
        assert re.fullmatch(r"\d{4}-\d{6}", pool.pension_number(rng).value)
        assert re.fullmatch(r"[A-Z]{2}\d{8}[A-Z]{2}", pool.residence_card(rng).value)
        acct = pool.bank_account(rng)
        if acct.meta.get("bank") == "ゆうちょ銀行":
            r = acct.render("ACCOUNT", "labeled")
            assert r.text.startswith("記号") and len(r.parts) == 2
        else:
            assert re.fullmatch(r"\d{7}", acct.value)
            r = acct.render("ACCOUNT", "labeled")
            assert r.text.endswith(acct.value) and len(r.parts) == 1 and r.text[r.parts[0].offset:] == acct.value
    styles = {pool.member_id(rng).value[:2] for _ in range(200)}
    assert len(styles) >= 3


# ---------------------------------------------------------------------------
# 骨格化
# ---------------------------------------------------------------------------
def test_to_skeleton_roundtrip_kinds():
    tpl = "氏名：{{PERSON_1}}（{{PERSON_KANA_1}}）TEL {{PHONE_1}} 〒 {{POSTAL_1}}\n住所 {{ADDRESS_1}} mail {{EMAIL_1}} {{SHARED_EMAIL_1}} {{DATE_BIRTH_1}} {{DATE_BUSINESS_1}} {{MYNUMBER_1}} {{URL_TRACK_1}} {{PUBLIC_URL_1}} {{SECRET_PASSWORD_1}} {{COMPANY_1}} {{AMOUNT_1}} {{PUBLIC_PERSON_1}} {{CORP_NUMBER_1}}"
    doc = _fill(tpl, "train", 12)
    sk = to_skeleton(doc.text, doc.entities, doc.keep_values)
    expected = re.sub(r"\{\{([A-Z_]+?)_\d+(?::[^}]*)?\}\}", r"{{\1}}", tpl)
    assert sk == expected
    # keep_values を渡さなければ KEEP(label なし) の値は残る
    sk2 = to_skeleton(doc.text, doc.entities)
    assert "{{COMPANY}}" not in sk2 and "{{PERSON}}" in sk2
    assert doc.skeleton() == sk


def test_name_element_classes_merge_homophones():
    from toramon.datagen.pools import SURNAME_CLASS, GIVEN_CLASS, person_combo_split
    assert SURNAME_CLASS["斎藤"] == SURNAME_CLASS["齋藤"] == SURNAME_CLASS["斉藤"]  # 同ローマ字
    assert SURNAME_CLASS["伊藤"] == SURNAME_CLASS["伊東"]  # 同読み
    assert SURNAME_CLASS["大野"] == SURNAME_CLASS["小野"]  # Ono
    assert SURNAME_CLASS["キム"] == SURNAME_CLASS["金"]  # 外国人名カタカナ/漢字
    assert SURNAME_CLASS["林"] == SURNAME_CLASS["林"] and SURNAME_CLASS["山田"] != SURNAME_CLASS["田中"]
    assert GIVEN_CLASS["孝之"] == GIVEN_CLASS["貴之"]
    # 同じ類の組み合わせは同じ split
    assert person_combo_split("斎藤", "孝之") == person_combo_split("齋藤", "貴之")
