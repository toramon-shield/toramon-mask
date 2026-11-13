"""合成値プール（Toramon Mask データ生成用）。

出所（provenance）
-----------------
- 本モジュールが返す値は**すべて合成値**である。氏名は姓・名の部品リスト（_data_names.py）を乱数で
  組み合わせたもので、特定の実在人物を意図しない。有名人・歴史上の人物は PUBLIC_PERSON（KEEP 用）の
  短いリストにのみ含む。
- 住所は実在の公的地名（都道府県・市区町村、_data_places.MUNICIPALITIES）に**架空の**町名・丁目・番地・
  建物名・部屋番号を組み合わせたもので、実在の完全な住所にはならない。郵便番号は形式のみ正しい架空値。
- 電話番号・メール・各種識別番号・URL・秘密情報は形式のみ模した乱数値。メールドメインと URL ホストは
  予約済み／試験用ドメイン（example.jp/.com/.net/.org、*.example、*.test、*.invalid）に限定する。
- マイナンバーと法人番号は公式のチェックデジット算法を実装し、正しい値／誤った値の両方を生成できる
  （どちらを生成したかは IdValue.meta["checksum_valid"] に記録）。

分割（pool_train / pool_dev / pool_test）
----------------------------------------
`Pool(split, seed)` は split ∈ {train, dev, test}。seed は乱数列のみを決め、分割の帰属は seed に依存しない
安定ハッシュ（sha256）で決める。したがって seed が違っても split 間で以下は互いに素:
- 姓×名の組み合わせ（表記・読み・ローマ字の同値類で判定するので、漢字/かな/ローマ字のどの描画でも重ならない。
  姓・名の要素単位の完全分離は求めない。ただし希少姓・希少名は要素単位でも分割し、
  test に専用の珍しい姓名を確保する）
- 市区町村×町名の組み合わせ
- メールのローカル部
- 各種識別番号の値（マイナンバー、免許証、口座番号、会員番号など）
比率: train 70% / dev 10% / test 20%（希少姓名は train 50% / dev 15% / test 35%）。

使い方
------
    pool = Pool("train", seed=1)
    p = pool.person()            # Person
    p.render("PERSON", "furigana").text   # '山田太郎（ヤマダタロウ）'
    pool.describe()              # KIND ごとの利用可能数
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import random
import string
from dataclasses import dataclass, field
from typing import Callable

from . import _data_names as N
from . import _data_places as P

SPLITS: tuple[str, ...] = ("train", "dev", "test")
SPLIT_RANGES: dict[str, tuple[int, int]] = {"train": (0, 70), "dev": (70, 80), "test": (80, 100)}
RARE_RANGES: dict[str, tuple[int, int]] = {"train": (0, 50), "dev": (50, 65), "test": (65, 100)}
_HASH_SALT = "toramon-pool-v1"


# ---------------------------------------------------------------------------
# 分割ハッシュ
# ---------------------------------------------------------------------------
def bucket(*parts: str) -> int:
    """安定ハッシュ → 0..99 のバケット。seed に依存しない。"""
    h = hashlib.sha256((_HASH_SALT + "|" + "|".join(parts)).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % 100


def split_of(*parts: str, ranges: dict[str, tuple[int, int]] = SPLIT_RANGES) -> str:
    b = bucket(*parts)
    for name, (lo, hi) in ranges.items():
        if lo <= b < hi:
            return name
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# 文字変換ユーティリティ
# ---------------------------------------------------------------------------
_FW_TABLE = {ord(c): chr(ord(c) - 0x30 + 0xFF10) for c in string.digits}
_FW_TABLE.update({ord("-"): "－", ord("("): "（", ord(")"): "）", ord("+"): "＋", ord("/"): "／", ord("."): "．", ord(" "): "　", ord(","): "，", ord("@"): "＠"})
_FW_ALPHA = {ord(c): chr(ord(c) - 0x41 + 0xFF21) for c in string.ascii_uppercase}
_FW_ALPHA.update({ord(c): chr(ord(c) - 0x61 + 0xFF41) for c in string.ascii_lowercase})


def to_fullwidth(s: str, alpha: bool = False) -> str:
    out = s.translate(_FW_TABLE)
    return out.translate(_FW_ALPHA) if alpha else out


def fw_digits(s: str) -> str:
    """数字と記号だけ全角化（文字は変えない）。"""
    return s.translate(_FW_TABLE)


def hira_to_kata(s: str) -> str:
    return "".join(chr(ord(c) + 0x60) if 0x3041 <= ord(c) <= 0x3096 else c for c in s)


def kata_to_hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in s)


def to_kyujitai(s: str) -> str:
    return "".join(N.KYUJITAI_MAP.get(c, c) for c in s)


def _ascii_token(s: str) -> str:
    return "".join(c for c in s.lower() if c in string.ascii_lowercase + string.digits)


# ---------------------------------------------------------------------------
# 描画結果（複数 span を持ち得る）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Part:
    """描画テキスト内の span 候補。role: primary=主実体, joint=連名の別人, furigana=同実体の読み。"""

    offset: int
    length: int
    role: str = "primary"
    subtype: str | None = None

    @property
    def end(self) -> int:
        return self.offset + self.length


@dataclass(frozen=True)
class Rendered:
    text: str
    parts: tuple[Part, ...]
    meta: dict = field(default_factory=dict, compare=False)

    @classmethod
    def whole(cls, text: str, subtype: str | None = None, **meta) -> "Rendered":
        return cls(text, (Part(0, len(text), "primary", subtype),), dict(meta))

    @classmethod
    def none(cls, text: str, **meta) -> "Rendered":
        """span を持たないテキスト（KEEP で entity を出さない種類）。"""
        return cls(text, (), dict(meta))


class VariantError(ValueError):
    pass


def _pick(variant: str | None, table: dict[str, Callable[[], Rendered]], default: str) -> Rendered:
    key = (variant or "").strip() or default
    if key == "default":
        key = default
    if key not in table:
        raise VariantError(f"unknown variant {variant!r}; allowed: {sorted(table)}")
    return table[key]()


# ---------------------------------------------------------------------------
# 氏名
# ---------------------------------------------------------------------------
@dataclass
class Person:
    surname: str
    given: str
    surname_kana: str  # ひらがな（外国人名はカタカナ）
    given_kana: str
    surname_romaji: str
    given_romaji: str
    gender: str = "u"
    kind: str = "ja"  # ja | foreign_kana | foreign_kanji
    rare: bool = False
    joint_given: tuple[str, str, str] | None = None  # (kanji, kana, romaji) 連名用の同姓の別人
    handle: str = ""

    # --- 基本形
    @property
    def is_foreign(self) -> bool:
        return self.kind != "ja"

    def full(self, sep: str = "") -> str:
        if self.kind == "foreign_kana":
            return self.given + (sep or "・") + self.surname
        return self.surname + sep + self.given

    def kana(self, katakana: bool = True, sep: str = "") -> str:
        if self.kind == "foreign_kana":
            return self.given_kana + (sep or "・") + self.surname_kana
        s, g = self.surname_kana, self.given_kana
        if self.kind == "ja" and not katakana:
            return s + sep + g
        return hira_to_kata(s) + sep + hira_to_kata(g)

    def romaji(self, style: str = "given_first") -> str:
        s, g = self.surname_romaji, self.given_romaji
        if self.kind == "foreign_kanji":
            # 漢字圏は姓→名が自然
            if style == "upper_family":
                return f"{s.upper()} {g}"
            if style == "upper_all":
                return f"{s.upper()} {g.upper()}"
            if style == "lower":
                return f"{s.lower()} {g.lower()}"
            if style == "initial":
                return f"{s} {g[0]}."
            return f"{s} {g}"
        if style == "given_first":
            return f"{g} {s}"
        if style == "family_first":
            return f"{s} {g}"
        if style == "upper_family":
            return f"{s.upper()} {g}"
        if style == "upper_all":
            return f"{g.upper()} {s.upper()}"
        if style == "lower":
            return f"{g.lower()} {s.lower()}"
        if style == "initial":
            return f"{g[0]}. {s}"
        raise VariantError(style)

    def ascii_tokens(self) -> tuple[str, str]:
        return _ascii_token(self.given_romaji), _ascii_token(self.surname_romaji)

    # --- 描画
    def _furigana(self, reading: str, sub: str, sep_main: str = "", sep_kana: str = "") -> Rendered:
        main = self.full(sep_main)
        text = f"{main}（{reading}）"
        return Rendered(text, (Part(0, len(main)), Part(len(main) + 1, len(reading), "furigana", sub)))

    def _joint(self, sep_main: str = "") -> Rendered:
        if not self.joint_given:
            return Rendered.whole(self.full(sep_main))
        main = self.full(sep_main)
        other = self.joint_given[0]
        text = f"{main}・{other}"
        return Rendered(text, (Part(0, len(main)), Part(len(main) + 1, len(other), "joint", "joint_name")))

    def render(self, kind: str = "PERSON", variant: str | None = None) -> Rendered:
        W = Rendered.whole
        if kind == "PERSON":
            table = {
                "nospace": lambda: W(self.full()),
                "space": lambda: W(self.full(" ")),
                "fullspace": lambda: W(self.full("　")),
                "family": lambda: W(self.surname, "family_name"),
                "given": lambda: W(self.given, "given_name"),
                "hiragana": lambda: W(self.kana(katakana=False), "furigana"),
                "katakana": lambda: W(self.kana(katakana=True), "furigana"),
                "kana_space": lambda: W(self.kana(True, " "), "furigana"),
                "romaji": lambda: W(self.romaji("given_first"), "romaji"),
                "romaji_upper": lambda: W(self.romaji("upper_family"), "romaji"),
                "romaji_sg": lambda: W(self.romaji("family_first"), "romaji"),
                "romaji_lower": lambda: W(self.romaji("lower"), "romaji"),
                "romaji_upper_all": lambda: W(self.romaji("upper_all"), "romaji"),
                "initial": lambda: W(self.romaji("initial"), "romaji"),
                "furigana": lambda: self._furigana(self.kana(True) if not self.is_foreign or self.kind == "foreign_kanji" else self.romaji(), "furigana" if self.kind != "foreign_kana" else "romaji"),
                "furigana_hira": lambda: self._furigana(self.kana(False) if self.kind == "ja" else self.kana(True), "furigana"),
                "furigana_space": lambda: self._furigana(self.kana(True, " "), "furigana", " ", " "),
                "kyujitai": lambda: W(to_kyujitai(self.full()), "kyujitai"),
                "kyujitai_space": lambda: W(to_kyujitai(self.full(" ")), "kyujitai"),
                "joint": lambda: self._joint(),
                "joint_space": lambda: self._joint(" "),
                "handle": lambda: W(self.handle, "handle"),
            }
            return _pick(variant, table, "nospace")
        if kind == "PERSON_KANA":
            table = {
                "katakana": lambda: W(self.kana(True), "furigana"),
                "hiragana": lambda: W(self.kana(False), "furigana"),
                "space": lambda: W(self.kana(True, " "), "furigana"),
                "fullspace": lambda: W(self.kana(True, "　"), "furigana"),
                "hiragana_space": lambda: W(self.kana(False, " "), "furigana"),
                "family": lambda: W(hira_to_kata(self.surname_kana), "furigana"),
                "given": lambda: W(hira_to_kata(self.given_kana), "furigana"),
                "family_hiragana": lambda: W(self.surname_kana if self.kind == "ja" else hira_to_kata(self.surname_kana), "furigana"),
            }
            return _pick(variant, table, "katakana")
        if kind == "PERSON_ROMAJI":
            table = {
                "given_first": lambda: W(self.romaji("given_first"), "romaji"),
                "upper": lambda: W(self.romaji("upper_family"), "romaji"),
                "sg": lambda: W(self.romaji("family_first"), "romaji"),
                "lower": lambda: W(self.romaji("lower"), "romaji"),
                "upper_all": lambda: W(self.romaji("upper_all"), "romaji"),
                "initial": lambda: W(self.romaji("initial"), "romaji"),
                "family": lambda: W(self.surname_romaji, "romaji"),
                "given": lambda: W(self.given_romaji, "romaji"),
            }
            return _pick(variant, table, "given_first")
        if kind == "PERSON_FAMILY":
            table = {
                "kanji": lambda: W(self.surname, "family_name"),
                "hiragana": lambda: W(self.surname_kana if self.kind == "ja" else hira_to_kata(self.surname_kana), "family_name"),
                "katakana": lambda: W(hira_to_kata(self.surname_kana), "family_name"),
                "romaji": lambda: W(self.surname_romaji, "family_name"),
                "romaji_upper": lambda: W(self.surname_romaji.upper(), "family_name"),
                "kyujitai": lambda: W(to_kyujitai(self.surname), "family_name"),
                "furigana": lambda: Rendered(f"{self.surname}（{hira_to_kata(self.surname_kana)}）", (Part(0, len(self.surname), "primary", "family_name"), Part(len(self.surname) + 1, len(self.surname_kana), "furigana", "furigana"))),
            }
            return _pick(variant, table, "kanji")
        if kind == "PERSON_GIVEN":
            table = {
                "kanji": lambda: W(self.given, "given_name"),
                "hiragana": lambda: W(self.given_kana if self.kind == "ja" else hira_to_kata(self.given_kana), "given_name"),
                "katakana": lambda: W(hira_to_kata(self.given_kana), "given_name"),
                "romaji": lambda: W(self.given_romaji, "given_name"),
            }
            return _pick(variant, table, "kanji")
        raise VariantError(f"unknown person kind {kind!r}")


# ---------------------------------------------------------------------------
# 住所
# ---------------------------------------------------------------------------
@dataclass
class Address:
    postal: str  # "221-0835"
    prefecture: str
    municipality: str
    town: str
    chome: int
    ban: int
    go: int | None
    building: str | None = None
    room: str | None = None
    has_building: bool = True
    corporate: bool = False  # COMPANY_ADDRESS 用

    def numbers(self, style: str = "hyphen") -> str:
        parts = [self.chome, self.ban] + ([self.go] if self.go is not None else [])
        if style == "hyphen":
            return "-".join(str(x) for x in parts)
        if style == "kanji":
            s = f"{self.chome}丁目{self.ban}番"
            return s + (f"{self.go}号" if self.go is not None else "")
        if style == "kanji_short":
            s = f"{self.chome}丁目{self.ban}"
            return s + (f"-{self.go}" if self.go is not None else "")
        raise VariantError(style)

    def street(self, style: str = "hyphen", with_pref: bool = True) -> str:
        head = (self.prefecture if with_pref else "") + self.municipality + self.town
        return head + self.numbers(style)

    def bldg(self, room_style: str = "plain") -> str:
        if not self.building:
            return ""
        if not self.room:
            return self.building
        if room_style == "plain":
            return f"{self.building}{self.room}"
        if room_style == "goshitsu":
            return f"{self.building} {self.room}号室"
        if room_style == "space":
            return f"{self.building} {self.room}"
        raise VariantError(room_style)

    def _full(self, postal: bool = True, num_style: str = "hyphen", building: bool | None = None, room_style: str = "plain", sep: str = " ", fullwidth: bool = False, with_pref: bool = True) -> str:
        use_b = self.has_building if building is None else building
        pieces = []
        if postal:
            pieces.append(f"〒{self.postal}")
        pieces.append(self.street(num_style, with_pref))
        if use_b and self.building:
            pieces.append(self.bldg(room_style))
        text = sep.join(pieces)
        return fw_digits(text) if fullwidth else text

    def render(self, kind: str = "ADDRESS", variant: str | None = None) -> Rendered:
        W = Rendered.whole
        if kind == "POSTAL":
            table = {
                "mark": lambda: W(f"〒{self.postal}", "postal_code_only"),
                "nomark": lambda: W(self.postal, "postal_code_only"),
                "fullwidth": lambda: W(fw_digits(f"〒{self.postal}"), "postal_code_only"),
                "nosep": lambda: W(f"〒{self.postal.replace('-', '')}", "postal_code_only"),
            }
            return _pick(variant, table, "mark")
        if kind == "ADDRESS_PARTIAL":
            table = {
                "municipality": lambda: W(self.prefecture + self.municipality, "partial"),
                "town": lambda: W(self.prefecture + self.municipality + self.town, "partial"),
                "nopref": lambda: W(self.municipality, "partial"),
                "prefecture": lambda: W(self.prefecture, "partial"),
            }
            return _pick(variant, table, "municipality")
        sub = "company" if self.corporate else None
        table = {
            "full": lambda: W(self._full(), sub),
            "nopostal": lambda: W(self._full(postal=False), sub),
            "nopostal_nobuilding": lambda: W(self._full(postal=False, building=False), sub),
            "nobuilding": lambda: W(self._full(building=False), sub),
            "building": lambda: W(self._full(building=True), sub),
            "room_goshitsu": lambda: W(self._full(building=True, room_style="goshitsu"), sub),
            "kanji_style": lambda: W(self._full(num_style="kanji"), sub),
            "kanji_style_nopostal": lambda: W(self._full(postal=False, num_style="kanji"), sub),
            "kanji_short": lambda: W(self._full(num_style="kanji_short"), sub),
            "fullwidth": lambda: W(self._full(fullwidth=True), sub),
            "fullwidth_nopostal": lambda: W(self._full(postal=False, fullwidth=True), sub),
            "nospace": lambda: W(self._full(sep=""), sub),
            "multiline": lambda: W(self._full(sep="\n"), sub),
            "multiline_nopostal": lambda: W(self._full(postal=False, sep="\n"), sub),
            "nopref": lambda: W(self._full(postal=False, with_pref=False), sub),
            "partial": lambda: W(self.prefecture + self.municipality, "partial"),
            "postal_only": lambda: W(f"〒{self.postal}", "postal_code_only"),
        }
        return _pick(variant, table, "full")


# ---------------------------------------------------------------------------
# 電話
# ---------------------------------------------------------------------------
# (市外局番, 種別) — 2〜5 桁。合計 10 桁になるよう市内局番を補う
AREA_CODES: tuple[tuple[str, str], ...] = (
    ("03", "landline"), ("06", "landline"), ("011", "landline"), ("022", "landline"), ("045", "landline"), ("048", "landline"),
    ("052", "landline"), ("075", "landline"), ("078", "landline"), ("082", "landline"), ("092", "landline"), ("098", "landline"),
    ("0422", "landline"), ("0467", "landline"), ("0476", "landline"), ("0797", "landline"), ("0263", "landline"), ("0742", "landline"),
    ("0155", "landline"), ("0985", "landline"), ("04", "landline"),
    ("090", "mobile"), ("080", "mobile"), ("070", "mobile"), ("050", "ip"),
)
COMPANY_PREFIXES: tuple[str, ...] = ("0120", "0570", "0800")


@dataclass
class Phone:
    area: str
    mid: str
    last: str
    kind: str = "landline"  # landline | mobile | ip | company
    ext: str | None = None

    def _sep(self, sep: str) -> str:
        return f"{self.area}{sep}{self.mid}{sep}{self.last}"

    def _intl(self, style: str) -> str:
        a = self.area[1:]
        if style == "hyphen":
            return f"+81-{a}-{self.mid}-{self.last}"
        if style == "space":
            return f"+81 {a} {self.mid} {self.last}"
        if style == "paren0":
            return f"+81 (0){a}-{self.mid}-{self.last}"
        if style == "nosep":
            return f"+81{a}{self.mid}{self.last}"
        raise VariantError(style)

    def render(self, kind: str = "PHONE", variant: str | None = None) -> Rendered:
        sub = {"PHONE": None, "PHONE_INTL": "intl", "FAX": "fax", "COMPANY_PHONE": "company_phone"}.get(kind)
        W = lambda t: Rendered.whole(t, sub)  # noqa: E731
        table = {
            "hyphen": lambda: W(self._sep("-")),
            "fullwidth": lambda: W(to_fullwidth(self._sep("-"))),
            "fullwidth_nosep": lambda: W(to_fullwidth(self._sep(""))),
            "paren": lambda: W(f"({self.area}){self.mid}-{self.last}"),
            "paren2": lambda: W(f"{self.area}({self.mid}){self.last}"),
            "paren_fullwidth": lambda: W(to_fullwidth(f"({self.area}){self.mid}-{self.last}")),
            "nosep": lambda: W(self._sep("")),
            "spaced": lambda: W(self._sep(" ")),
            "dot": lambda: W(self._sep(".")),
            "intl": lambda: W(self._intl("hyphen")),
            "intl_space": lambda: W(self._intl("space")),
            "intl_paren0": lambda: W(self._intl("paren0")),
            "intl_nosep": lambda: W(self._intl("nosep")),
            "ext": lambda: W(f"{self._sep('-')} 内線{self.ext or '204'}"),
            "ext_paren": lambda: W(f"{self._sep('-')}（内線{self.ext or '204'}）"),
        }
        return _pick(variant, table, "intl" if kind == "PHONE_INTL" else "hyphen")


# ---------------------------------------------------------------------------
# メール / URL / 文字列値
# ---------------------------------------------------------------------------
@dataclass
class Email:
    local: str
    domain: str
    shared: bool = False

    @property
    def text(self) -> str:
        return f"{self.local}@{self.domain}"

    def render(self, kind: str = "EMAIL", variant: str | None = None) -> Rendered:
        sub = "shared" if self.shared else None
        table = {
            "plain": lambda: Rendered.whole(self.text, sub),
            "fullwidth_at": lambda: Rendered.whole(f"{self.local}＠{self.domain}", sub),
            "upper": lambda: Rendered.whole(self.text.upper(), sub),
            "fullwidth": lambda: Rendered.whole(to_fullwidth(self.text, alpha=True), sub),
            "mailto": lambda: Rendered(f"mailto:{self.text}", (Part(7, len(self.text), "primary", sub),)),
        }
        return _pick(variant, table, "plain")


@dataclass
class Simple:
    """単一文字列の値（URL・秘密情報・KEEP 系など）。"""

    text: str
    subtype: str | None = None
    meta: dict = field(default_factory=dict)
    emit_span: bool = True

    def render(self, kind: str = "", variant: str | None = None) -> Rendered:
        mk = (lambda t: Rendered.whole(t, self.subtype, **self.meta)) if self.emit_span else (lambda t: Rendered.none(t, **self.meta))
        table = {
            "plain": lambda: mk(self.text),
            "fullwidth": lambda: mk(to_fullwidth(self.text, alpha=True)),
            "fullwidth_digits": lambda: mk(fw_digits(self.text)),
            "noscheme": lambda: mk(self.text.split("://", 1)[-1]),
            "bearer": lambda: Rendered(f"Bearer {self.text}", (Part(7, len(self.text), "primary", self.subtype),), dict(self.meta)),
            "upper": lambda: mk(self.text.upper()),
            "lower": lambda: mk(self.text.lower()),
        }
        return _pick(variant, table, "plain")


@dataclass
class ConnString:
    """接続文字列。span はパスワード部分のみ。"""

    scheme: str
    user: str
    password: str
    host: str
    port: int
    db: str

    def render(self, kind: str = "SECRET_CONNSTR", variant: str | None = None) -> Rendered:
        def uri() -> Rendered:
            pre = f"{self.scheme}://{self.user}:"
            text = f"{pre}{self.password}@{self.host}:{self.port}/{self.db}"
            return Rendered(text, (Part(len(pre), len(self.password), "primary", "connection_string"),))

        def kv() -> Rendered:
            pre = f"Server={self.host},{self.port};Database={self.db};User Id={self.user};Password="
            text = f"{pre}{self.password};"
            return Rendered(text, (Part(len(pre), len(self.password), "primary", "connection_string"),))

        def jdbc() -> Rendered:
            pre = f"jdbc:{self.scheme}://{self.host}:{self.port}/{self.db}?user={self.user}&password="
            text = pre + self.password
            return Rendered(text, (Part(len(pre), len(self.password), "primary", "connection_string"),))

        return _pick(variant, {"uri": uri, "kv": kv, "jdbc": jdbc}, "uri")


# ---------------------------------------------------------------------------
# 識別番号
# ---------------------------------------------------------------------------
def my_number_check_digit(body11: str) -> int:
    """マイナンバー検査用数字（行政手続における特定の個人を識別するための番号の利用等に関する法律施行令 第8条）。

    P_n: 右から n 桁目（n=1..11）、Q_n = n+1 (n<=6) / n-5 (n>=7)。
    検査用数字 = 11 - (Σ P_n×Q_n mod 11)。ただし mod が 1 以下なら 0。
    """
    if len(body11) != 11 or not body11.isdigit():
        raise ValueError("body must be 11 digits")
    total = 0
    for n in range(1, 12):
        p = int(body11[-n])
        q = n + 1 if n <= 6 else n - 5
        total += p * q
    r = total % 11
    return 0 if r <= 1 else 11 - r


def is_valid_my_number(value: str) -> bool:
    digits = "".join(c for c in value if c.isdigit())
    return len(digits) == 12 and my_number_check_digit(digits[:11]) == int(digits[11])


def corp_number_check_digit(body12: str) -> int:
    """法人番号の検査用数字（国税庁: 9 - (Σ P_n×Q_n mod 9)、Q_n は右から奇数桁 1 / 偶数桁 2）。先頭に付く。"""
    if len(body12) != 12 or not body12.isdigit():
        raise ValueError("body must be 12 digits")
    total = 0
    for n in range(1, 13):
        p = int(body12[-n])
        q = 1 if n % 2 == 1 else 2
        total += p * q
    return 9 - (total % 9)


def is_valid_corp_number(value: str) -> bool:
    digits = "".join(c for c in value if c.isdigit())
    return len(digits) == 13 and corp_number_check_digit(digits[1:]) == int(digits[0])


@dataclass
class IdValue:
    value: str
    subtype: str
    meta: dict = field(default_factory=dict)
    labeled: tuple[tuple[str, str], ...] = ()  # ((label, value), ...) 記号・番号など複数部分
    keep: bool = False

    def _labeled(self, sep: str = " ") -> Rendered:
        pieces, parts, pos = [], [], 0
        for i, (lab, val) in enumerate(self.labeled):
            if i:
                pieces.append(sep)
                pos += len(sep)
            pieces.append(lab)
            pos += len(lab)
            pieces.append(val)
            parts.append(Part(pos, len(val), "primary", self.subtype))
            pos += len(val)
        return Rendered("".join(pieces), tuple(parts), dict(self.meta))

    def render(self, kind: str = "", variant: str | None = None) -> Rendered:
        W = lambda t: Rendered.whole(t, self.subtype, **self.meta)  # noqa: E731
        digits_only = "".join(c for c in self.value if c.isalnum())
        table = {
            "plain": lambda: W(self.value),
            "fullwidth": lambda: W(to_fullwidth(self.value, alpha=True)),
            "nosep": lambda: W(digits_only),
            "spaced4": lambda: W(" ".join(digits_only[i : i + 4] for i in range(0, len(digits_only), 4))),
            "hyphen4": lambda: W("-".join(digits_only[i : i + 4] for i in range(0, len(digits_only), 4))),
            "labeled": lambda: self._labeled(" ") if self.labeled else W(self.value),
            "labeled_fullwidth": lambda: self._labeled("　") if self.labeled else W(self.value),
            "lower": lambda: W(self.value.lower()),
        }
        return _pick(variant, table, "plain")


# ---------------------------------------------------------------------------
# 日付
# ---------------------------------------------------------------------------
_ERAS: tuple[tuple[str, str, _dt.date], ...] = (
    ("令和", "R", _dt.date(2019, 5, 1)),
    ("平成", "H", _dt.date(1989, 1, 8)),
    ("昭和", "S", _dt.date(1926, 12, 25)),
    ("大正", "T", _dt.date(1912, 7, 30)),
    ("明治", "M", _dt.date(1868, 1, 25)),
)
_WEEKDAYS = "月火水木金土日"


@dataclass
class DateVal:
    date: _dt.date
    subtype: str | None = None

    def wareki(self) -> tuple[str, str, int]:
        for name, letter, start in _ERAS:
            if self.date >= start:
                return name, letter, self.date.year - start.year + 1
        raise ValueError("date before Meiji")

    def render(self, kind: str = "DATE_PERSONAL", variant: str | None = None) -> Rendered:
        d = self.date
        W = lambda t: Rendered.whole(t, self.subtype)  # noqa: E731
        era, letter, ey = self.wareki()
        ey_s = "元" if ey == 1 else str(ey)
        wd = _WEEKDAYS[d.weekday()]
        table = {
            "kanji": lambda: W(f"{d.year}年{d.month}月{d.day}日"),
            "kanji_zero": lambda: W(f"{d.year}年{d.month:02d}月{d.day:02d}日"),
            "slash": lambda: W(f"{d.year}/{d.month:02d}/{d.day:02d}"),
            "slash_short": lambda: W(f"{d.year}/{d.month}/{d.day}"),
            "iso": lambda: W(f"{d.year}-{d.month:02d}-{d.day:02d}"),
            "dot": lambda: W(f"{d.year}.{d.month}.{d.day}"),
            "dot_short": lambda: W(f"{d.year % 100}.{d.month}.{d.day}"),
            "compact": lambda: W(f"{d.year}{d.month:02d}{d.day:02d}"),
            "wareki": lambda: W(f"{era}{ey_s}年{d.month}月{d.day}日"),
            "wareki_short": lambda: W(f"{letter}{ey}.{d.month}.{d.day}"),
            "wareki_short_zero": lambda: W(f"{letter}{ey:02d}.{d.month:02d}.{d.day:02d}"),
            "wareki_year": lambda: W(f"{era}{ey_s}年"),
            "wareki_weekday": lambda: W(f"{era}{ey_s}年{d.month}月{d.day}日（{wd}）"),
            "weekday": lambda: W(f"{d.year}年{d.month}月{d.day}日（{wd}）"),
            "weekday_slash": lambda: W(f"{d.year}/{d.month}/{d.day}({wd})"),
            "weekday_kanji": lambda: W(f"{d.year}年{d.month}月{d.day}日{wd}曜日"),
            "fullwidth": lambda: W(fw_digits(f"{d.year}年{d.month}月{d.day}日")),
            "fullwidth_slash": lambda: W(to_fullwidth(f"{d.year}/{d.month:02d}/{d.day:02d}")),
            "fullwidth_wareki": lambda: W(fw_digits(f"{era}{ey_s}年{d.month}月{d.day}日")),
            "month_day": lambda: W(f"{d.month}月{d.day}日"),
            "year_month": lambda: W(f"{d.year}年{d.month}月"),
            "english": lambda: W(d.strftime("%B %-d, %Y")),
        }
        return _pick(variant, table, "kanji")


# ---------------------------------------------------------------------------
# 氏名要素の同値類（表記・読み・ローマ字のいずれかが同じなら同じ類）
# ---------------------------------------------------------------------------
def _element_classes(entries: list[tuple[str, str, str]]) -> dict[str, str]:
    """(表記, かな[ひらがな正規化], ローマ字) の列から、表記 → 類 ID を作る。

    同じ表記（林/林）、同じ読み（伊藤/伊東）、同じローマ字（斎藤/齋藤 → Saito、大野/小野 → Ono）は 1 つの類に
    まとめる。姓×名の分割判定を類 ID で行うことで、漢字・かな・ローマ字のどの表記で描画しても split 間で重ならない。
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for written, kana, romaji in entries:
        w = "w:" + written
        union(w, "k:" + kata_to_hira(kana))
        union(w, "r:" + _ascii_token(romaji))
    return {e[0]: find("w:" + e[0]) for e in entries}


SURNAME_CLASS: dict[str, str] = _element_classes(
    [(s[0], s[1], s[2]) for s in N.SURNAMES_COMMON + N.SURNAMES_RARE]
    + [(s[0], s[0], s[1]) for s in N.FOREIGN_SURNAMES]
    + [(s[0], s[1], s[2]) for s in N.KANJI_FOREIGN_SURNAMES]
)
GIVEN_CLASS: dict[str, str] = _element_classes(
    [(g[0], g[1], g[2]) for g in N.GIVEN_COMMON + N.GIVEN_RARE]
    + [(g[0], g[0], g[1]) for g in N.FOREIGN_GIVEN]
    + [(g[0], g[1], g[2]) for g in N.KANJI_FOREIGN_GIVEN]
)


def person_combo_split(surname_written: str, given_written: str) -> str:
    """姓・名（表記）の組み合わせが属する split。"""
    return split_of("person_class", SURNAME_CLASS[surname_written], GIVEN_CLASS[given_written])


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------
class Pool:
    """split 専属の合成値プール。全メソッドは `rng` を省略すると内部の seeded RNG を使う。"""

    def __init__(self, split: str, seed: int = 0, foreign_ratio: float = 0.10, rare_ratio: float = 0.15, max_tries: int = 20000):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.split = split
        self.seed = seed
        self.foreign_ratio = foreign_ratio
        self.rare_ratio = rare_ratio
        self.max_tries = max_tries
        self.rng = random.Random(f"{split}:{seed}")
        # 希少姓名は要素単位で分割（test に専用の珍しい姓名を確保）
        self._rare_surnames = tuple(s for s in N.SURNAMES_RARE if split_of("rare_surname", s[0], s[1], ranges=RARE_RANGES) == split)
        self._rare_given = tuple(g for g in N.GIVEN_RARE if split_of("rare_given", g[0], g[1], ranges=RARE_RANGES) == split)
        self._common_surnames = N.SURNAMES_COMMON
        self._given = N.GIVEN_COMMON + self._rare_given

    # --- 内部
    def _rng(self, rng: random.Random | None) -> random.Random:
        return rng if rng is not None else self.rng

    def in_split(self, *key: str) -> bool:
        return split_of(*key) == self.split

    def _until(self, rng: random.Random, make: Callable[[], str], ns: str) -> str:
        for _ in range(self.max_tries):
            v = make()
            if self.in_split(ns, v):
                return v
        raise RuntimeError(f"could not draw a {ns} value for split {self.split}")

    @staticmethod
    def _digits(rng: random.Random, n: int, first_nonzero: bool = False) -> str:
        s = "".join(rng.choice(string.digits) for _ in range(n))
        if first_nonzero and s[0] == "0":
            s = rng.choice("123456789") + s[1:]
        return s

    # --- 氏名
    def _combo_ok(self, surname_written: str, given_written: str) -> bool:
        return person_combo_split(surname_written, given_written) == self.split

    def _ja_combo_ok(self, surname: tuple[str, str, str], given: tuple[str, str, str, str]) -> bool:
        return self._combo_ok(surname[0], given[0])

    def person(self, rng: random.Random | None = None, *, foreign: bool | None = None, gender: str | None = None, rare: bool | None = None, kyujitai: bool = False) -> Person:
        """合成人物を 1 人引く。kyujitai=True なら旧字体へ写せる文字を含む氏名（日本人名）を選ぶ。"""
        rng = self._rng(rng)
        if kyujitai:
            foreign = False
        if foreign is None:
            foreign = rng.random() < self.foreign_ratio
        for _ in range(self.max_tries):
            if foreign:
                if rng.random() < 0.8:
                    g = rng.choice(N.FOREIGN_GIVEN)
                    if gender and g[2] not in (gender, "u"):
                        continue
                    s = rng.choice(N.FOREIGN_SURNAMES)
                    if not self._combo_ok(s[0], g[0]):
                        continue
                    p = Person(s[0], g[0], s[0], g[0], s[1], g[1], g[2], "foreign_kana", False)
                else:
                    g2 = rng.choice(N.KANJI_FOREIGN_GIVEN)
                    if gender and g2[3] not in (gender, "u"):
                        continue
                    s2 = rng.choice(N.KANJI_FOREIGN_SURNAMES)
                    if not self._combo_ok(s2[0], g2[0]):
                        continue
                    p = Person(s2[0], g2[0], s2[1], g2[1], s2[2], g2[2], g2[3], "foreign_kanji", False)
                p.handle = self._handle(rng, p)
                return p
            use_rare = (rng.random() < self.rare_ratio) if rare is None else rare
            if use_rare and self._rare_surnames and rng.random() < 0.7:
                s = rng.choice(self._rare_surnames)
                is_rare = True
            else:
                s = rng.choice(self._common_surnames)
                is_rare = False
            givens = self._rare_given if (use_rare and not is_rare and self._rare_given) else self._given
            g = rng.choice(givens)
            if gender and g[3] not in (gender, "u"):
                continue
            if not self._ja_combo_ok(s, g):
                continue
            if kyujitai and to_kyujitai(s[0] + g[0]) == s[0] + g[0]:
                continue
            p = Person(s[0], g[0], s[1], g[1], s[2], g[2], g[3], "ja", is_rare or g in self._rare_given)
            p.joint_given = self._joint_given(rng, s, g)
            p.handle = self._handle(rng, p)
            return p
        raise RuntimeError(f"could not draw a person for split {self.split}")

    def _joint_given(self, rng: random.Random, s: tuple[str, str, str], g: tuple[str, str, str, str]) -> tuple[str, str, str] | None:
        want = {"m": "f", "f": "m"}.get(g[3])
        for _ in range(200):
            o = rng.choice(self._given)
            if o[0] == g[0] or (want and o[3] not in (want, "u")):
                continue
            if self._ja_combo_ok(s, o):
                return (o[0], o[1], o[2])
        return None

    def _handle(self, rng: random.Random, p: Person) -> str:
        g, s = p.ascii_tokens()
        sep = rng.choice(["_", ".", "", "-", "_"])
        num = rng.choice(["", "", str(rng.randint(1, 99)), str(rng.randint(1960, 2010)), str(rng.randint(100, 9999))])
        style = rng.choice([f"{g}{sep}{s}{num}", f"{s}{sep}{g}{num}", f"{g[0]}{sep}{s}{num}", f"{g}{num}"])
        return style

    def spouse(self, person: Person, rng: random.Random | None = None) -> Person:
        """同姓の別人（連名・家族）。"""
        rng = self._rng(rng)
        if person.joint_given:
            k, kana, ro = person.joint_given
            gender = next((g[3] for g in self._given if g[0] == k and g[1] == kana), "u")
            q = Person(person.surname, k, person.surname_kana, kana, person.surname_romaji, ro, gender, person.kind, person.rare)
            q.handle = self._handle(rng, q)
            return q
        return self.person(rng)

    # --- 住所
    def address(self, rng: random.Random | None = None, *, corporate: bool = False) -> Address:
        rng = self._rng(rng)
        for _ in range(self.max_tries):
            pref, muni = rng.choice(P.MUNICIPALITIES)
            town = rng.choice(P.TOWNS)
            if not self.in_split("address", muni, town):
                continue
            postal = f"{rng.randint(100, 999):03d}-{rng.randint(0, 9999):04d}"
            chome = rng.randint(1, 6)
            ban = rng.randint(1, 30) if rng.random() < 0.7 else rng.randint(31, 99)
            go = rng.randint(1, 25) if rng.random() < 0.8 else None
            if corporate:
                bld = rng.choice(P.OFFICE_BUILDINGS)
                room = f"{rng.randint(2, 12)}F"
                return Address(postal, pref, muni, town, chome, ban, go, bld, room, True, True)
            bld = f"{rng.choice(P.BUILDING_TYPES)}{town.rstrip('町台野')}{rng.choice(P.BUILDING_SUFFIX)}"
            room = str(rng.randint(1, 15) * 100 + rng.randint(1, 12)) if rng.random() < 0.85 else None
            return Address(postal, pref, muni, town, chome, ban, go, bld, room, rng.random() < 0.6)
        raise RuntimeError("could not draw an address")

    def company_address(self, rng: random.Random | None = None) -> Address:
        return self.address(rng, corporate=True)

    # --- 電話
    def phone(self, rng: random.Random | None = None, *, kind: str | None = None) -> Phone:
        rng = self._rng(rng)
        if kind == "company":
            return self.company_phone(rng)
        for _ in range(self.max_tries):
            candidates = [a for a in AREA_CODES if kind is None or a[1] == kind]
            area, k = rng.choice(candidates)
            rest = 10 - len(area)  # 市内局番 + 加入者番号(4)
            if k != "landline":
                rest = 11 - len(area)
            mid = self._digits(rng, rest - 4, first_nonzero=True)
            last = self._digits(rng, 4)
            if self.in_split("phone", area + mid + last):
                return Phone(area, mid, last, k, ext=str(rng.randint(100, 999)))
        raise RuntimeError("could not draw a phone")

    def mobile(self, rng: random.Random | None = None) -> Phone:
        return self.phone(rng, kind="mobile")

    def fax(self, rng: random.Random | None = None) -> Phone:
        return self.phone(rng, kind="landline")

    def company_phone(self, rng: random.Random | None = None) -> Phone:
        rng = self._rng(rng)
        for _ in range(self.max_tries):
            area = rng.choice(COMPANY_PREFIXES)
            mid = self._digits(rng, 3, first_nonzero=True)
            last = self._digits(rng, 4 if area == "0800" else 3)
            if self.in_split("phone", area + mid + last):
                return Phone(area, mid, last, "company")
        raise RuntimeError("could not draw a company phone")

    # --- メール
    def email(self, rng: random.Random | None = None, *, person: Person | None = None) -> Email:
        rng = self._rng(rng)
        person = person or self.person(rng)
        g, s = person.ascii_tokens()
        yy = str(rng.randint(60, 99)) if rng.random() < 0.5 else str(rng.randint(1970, 2012))
        styles = [f"{g}.{s}", f"{g[0]}.{s}", f"{s}.{g}", f"{g}_{s}", f"{g}-{s}", f"{s}{g[0]}", f"{g}{s}", f"{g}.{s}{yy[-2:]}", f"{s}.{g[0]}", f"{g}{yy}", f"{s}_{g}{rng.randint(1, 99)}", f"{g[0]}{s}{rng.randint(1, 999)}"]
        rng.shuffle(styles)
        domain = rng.choice(P.EMAIL_DOMAINS)
        for local in styles:
            if self.in_split("email_local", local):
                return Email(local, domain)
        base = styles[0]
        for i in range(1, self.max_tries):
            local = f"{base}{rng.randint(1, 9999)}"
            if self.in_split("email_local", local):
                return Email(local, domain)
        raise RuntimeError("could not draw an email")

    def shared_email(self, rng: random.Random | None = None, *, domain: str | None = None) -> Email:
        rng = self._rng(rng)
        local = rng.choice(["info", "support", "sales", "contact", "inquiry", "cs", "help", "noreply", "no-reply", "admin", "recruit", "press", "order", "billing", "webmaster"])
        return Email(local, domain or rng.choice(P.COMPANY_DOMAINS), shared=True)

    # --- 識別番号（account_number）
    def my_number(self, rng: random.Random | None = None, *, valid: bool = True) -> IdValue:
        rng = self._rng(rng)

        def make() -> str:
            body = self._digits(rng, 11)
            cd = my_number_check_digit(body)
            if not valid:
                cd = (cd + rng.randint(1, 9)) % 10
            return body + str(cd)

        v = self._until(rng, make, "mynumber")
        return IdValue(v, "my_number", {"checksum_valid": valid})

    def corp_number(self, rng: random.Random | None = None) -> IdValue:
        rng = self._rng(rng)
        body = self._digits(rng, 12, first_nonzero=True)
        return IdValue(str(corp_number_check_digit(body)) + body, "corp_number", {"checksum_valid": True}, keep=True)

    def license_number(self, rng: random.Random | None = None) -> IdValue:
        rng = self._rng(rng)
        v = self._until(rng, lambda: f"{rng.randint(10, 97):02d}{rng.randint(80, 125) % 100:02d}{self._digits(rng, 6)}{rng.randint(0, 9)}{rng.randint(0, 2)}", "license")
        return IdValue(v, "drivers_license")

    def passport_number(self, rng: random.Random | None = None) -> IdValue:
        rng = self._rng(rng)
        v = self._until(rng, lambda: rng.choice(["TK", "TR", "TZ", "MT", "ME", "MZ", "XS", "TH"]) + self._digits(rng, 7), "passport")
        return IdValue(v, "passport")

    def pension_number(self, rng: random.Random | None = None) -> IdValue:
        rng = self._rng(rng)
        v = self._until(rng, lambda: f"{self._digits(rng, 4, True)}-{self._digits(rng, 6)}", "pension")
        return IdValue(v, "pension")

    def insurance_number(self, rng: random.Random | None = None) -> IdValue:
        rng = self._rng(rng)
        kigo = self._until(rng, lambda: rng.choice([self._digits(rng, 8, True), self._digits(rng, 4, True), f"{rng.randint(1, 99)}-{self._digits(rng, 4)}", f"{rng.choice(['公', '協', '共'])}{self._digits(rng, 5)}"]), "insurance")
        bango = str(rng.randint(1, 999))
        eda = f"{rng.randint(0, 3):02d}"
        labeled = (("記号", kigo), ("番号", bango))
        if rng.random() < 0.3:
            labeled = labeled + (("枝番", eda),)
        return IdValue(f"{kigo}-{bango}", "health_insurance", labeled=labeled)

    def bank_account(self, rng: random.Random | None = None) -> IdValue:
        rng = self._rng(rng)
        if rng.random() < 0.25:
            kigo = self._until(rng, lambda: "1" + self._digits(rng, 3) + "0", "bank")
            bango = self._digits(rng, 7, True) + "1"
            return IdValue(f"{kigo}-{bango}", "bank_account", {"bank": "ゆうちょ銀行"}, labeled=(("記号", kigo), ("番号", bango)))
        num = self._until(rng, lambda: self._digits(rng, 7, True), "bank")
        bank, branch = rng.choice(P.BANKS), rng.choice(P.BRANCHES)
        acct_type = rng.choice(["普通", "普通", "当座", "貯蓄"])
        return IdValue(num, "bank_account", {"bank": bank, "branch": branch, "type": acct_type}, labeled=((f"{bank} {branch} {acct_type} ", num),))

    def residence_card(self, rng: random.Random | None = None) -> IdValue:
        rng = self._rng(rng)
        L = string.ascii_uppercase
        v = self._until(rng, lambda: rng.choice(L) + rng.choice(L) + self._digits(rng, 8) + rng.choice(L) + rng.choice(L), "residence")
        return IdValue(v, "residence_card")

    def _styled_id(self, rng: random.Random, ns: str, styles: list[Callable[[], str]], subtype: str) -> IdValue:
        v = self._until(rng, lambda: rng.choice(styles)(), ns)
        return IdValue(v, subtype)

    def member_id(self, rng: random.Random | None = None) -> IdValue:
        rng = self._rng(rng)
        y = rng.randint(2018, 2027)
        return self._styled_id(rng, "member", [
            lambda: f"M-{self._digits(rng, 6, True)}", lambda: f"MB-{y}-{self._digits(rng, 5)}", lambda: f"KS{self._digits(rng, 9)}",
            lambda: f"{self._digits(rng, 8, True)}", lambda: f"MEM{self._digits(rng, 7)}", lambda: f"{rng.choice('ABCD')}{self._digits(rng, 3)}-{self._digits(rng, 4)}",
        ], "member_id")

    def customer_id(self, rng: random.Random | None = None) -> IdValue:
        rng = self._rng(rng)
        return self._styled_id(rng, "customer", [
            lambda: f"C{self._digits(rng, 7, True)}", lambda: f"CUST-{self._digits(rng, 6)}", lambda: f"CS-{self._digits(rng, 4)}-{self._digits(rng, 4)}",
            lambda: f"{self._digits(rng, 10, True)}", lambda: f"U{self._digits(rng, 8)}",
        ], "customer_id")

    def contract_id(self, rng: random.Random | None = None) -> IdValue:
        rng = self._rng(rng)
        y = rng.randint(2018, 2027)
        return self._styled_id(rng, "contract", [
            lambda: f"K-{y}-{self._digits(rng, 4)}", lambda: f"CT{y}{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}-{self._digits(rng, 4)}",
            lambda: f"{self._digits(rng, 4, True)}-{self._digits(rng, 6)}", lambda: f"CN{self._digits(rng, 9)}",
        ], "contract_id")

    def order_id(self, rng: random.Random | None = None) -> IdValue:
        rng = self._rng(rng)
        y = rng.randint(2024, 2027)
        m, d = rng.randint(1, 12), rng.randint(1, 28)
        return self._styled_id(rng, "order", [
            lambda: f"R{y}{m:02d}{d:02d}-{self._digits(rng, 4)}", lambda: f"ORD-{y}-{self._digits(rng, 6)}", lambda: f"{y}-{self._digits(rng, 5)}",
            lambda: f"{y}{m:02d}{d:02d}-{self._digits(rng, 4)}-{rng.randint(1, 9):02d}", lambda: f"#{self._digits(rng, 8, True)}", lambda: f"{self._digits(rng, 3, True)}-{self._digits(rng, 7)}-{self._digits(rng, 7)}",
        ], "order_id")

    def employee_id(self, rng: random.Random | None = None) -> IdValue:
        rng = self._rng(rng)
        return self._styled_id(rng, "employee", [
            lambda: f"E{self._digits(rng, 5, True)}", lambda: f"EMP-{self._digits(rng, 5)}", lambda: f"{self._digits(rng, 6, True)}", lambda: f"S{rng.randint(90, 127) % 100:02d}{self._digits(rng, 4)}",
        ], "employee_id")

    # --- 日付
    def _date(self, rng: random.Random, y0: int, y1: int, subtype: str) -> DateVal:
        y = rng.randint(y0, y1)
        m = rng.randint(1, 12)
        d = rng.randint(1, 28 if m == 2 else 30 if m in (4, 6, 9, 11) else 31)
        return DateVal(_dt.date(y, m, d), subtype)

    def date_birth(self, rng: random.Random | None = None) -> DateVal:
        return self._date(self._rng(rng), 1938, 2008, "birth")

    def date_personal(self, rng: random.Random | None = None) -> DateVal:
        return self._date(self._rng(rng), 2012, 2027, "personal_event")

    def date_business(self, rng: random.Random | None = None) -> DateVal:
        return self._date(self._rng(rng), 2023, 2027, "business")

    # --- URL
    def _token(self, rng: random.Random, n: int, alphabet: str = string.ascii_letters + string.digits) -> str:
        return "".join(rng.choice(alphabet) for _ in range(n))

    def url_track(self, rng: random.Random | None = None) -> Simple:
        rng = self._rng(rng)
        hosts = ["track.example.test", "delivery.example.jp", "trace.suiha-logi.example", "t.example.net", "tr.example.invalid"]
        tok = self._token(rng, rng.randint(6, 10), string.ascii_lowercase + string.digits)
        styles = [
            lambda: f"https://{rng.choice(hosts)}/p/{tok}",
            lambda: f"https://{rng.choice(hosts)}/track?no={self._digits(rng, 4)}-{self._digits(rng, 4)}-{self._digits(rng, 4)}",
            lambda: f"https://{rng.choice(hosts)}/s/{self._digits(rng, 12)}",
            lambda: f"http://{rng.choice(hosts)}/t/{tok}?lang=ja",
        ]
        v = self._until(rng, lambda: rng.choice(styles)(), "url")
        return Simple(v, "tracking")

    def url_personal(self, rng: random.Random | None = None, *, person: Person | None = None) -> Simple:
        rng = self._rng(rng)
        handle = person.handle if person else self._handle(rng, self.person(rng))
        tok = self._token(rng, rng.randint(20, 40), string.ascii_letters + string.digits + "-_")
        styles = [
            lambda: f"https://portal.example.jp/reset?token={tok}",
            lambda: f"https://sns.example.com/u/{handle}",
            lambda: f"https://sns.example.net/@{handle}",
            lambda: f"https://drive.example.net/s/{self._token(rng, 10)}",
            lambda: f"https://my.example.org/invite/{tok[:16]}",
            lambda: f"https://{rng.choice(P.COMPANY_DOMAINS)}/mypage?uid={self._digits(rng, 8)}&sig={tok[:12]}",
            lambda: f"https://meet.example.test/j/{self._digits(rng, 10)}?pwd={tok[:10]}",
            lambda: f"https://blog.example.jp/{handle}/",
        ]
        v = self._until(rng, lambda: rng.choice(styles)(), "url")
        return Simple(v, "personal")

    def public_url(self, rng: random.Random | None = None) -> Simple:
        rng = self._rng(rng)
        dom = rng.choice(P.COMPANY_DOMAINS)
        slug = rng.choice(["products", "news", "company", "support/faq", "docs/guide", "recruit", "contact", "privacy", "downloads", "ir"])
        tail = rng.choice(["", "/", f"/{rng.choice(P.PRODUCT_PREFIXES).lower()}-{rng.randint(100, 9999)}", "/index.html", f"/{rng.randint(2024, 2027)}/"])
        v = rng.choice([f"https://www.{dom}/{slug}{tail}", f"https://{dom}/{slug}{tail}", f"https://{dom}/", f"https://docs.{dom}/{slug}{tail}"])
        return Simple(v, "public")

    # --- 秘密情報
    def password(self, rng: random.Random | None = None, *, person: Person | None = None) -> Simple:
        rng = self._rng(rng)
        g, s = (person.ascii_tokens() if person else self.person(rng).ascii_tokens())
        kana_bits = ["ゆうこ", "たろう", "さくら", "パスワード", "ひみつ", "ねこ", "いぬ", "こうべ"]
        sym = rng.choice("!#$%&*?@_-")
        year = rng.randint(1960, 2026)
        styles = [
            lambda: f"{g.capitalize()}{year}{sym}",
            lambda: f"{s}{rng.randint(1, 9999):04d}",
            lambda: self._token(rng, rng.randint(8, 16), string.ascii_letters + string.digits + "!#$%&*?@_-"),
            lambda: f"{rng.choice(kana_bits)}{rng.randint(1, 999)}{sym}",
            lambda: f"{rng.choice(['P@ssw0rd', 'Passw0rd', 'pass', 'Pr1v4cy', 'Welcome'])}{sym}{rng.randint(1, 99)}",
            lambda: f"{g}{sym}{s}{rng.randint(1, 99)}",
            lambda: f"{rng.choice(['Sakura', 'Fuji', 'Tokyo', 'Kaede', 'Hikari'])}{rng.randint(1, 9999)}{sym}{rng.choice(['x', 'Z', 'q'])}",
            lambda: self._token(rng, 6, string.digits),
            lambda: f"{rng.choice(kana_bits)}_{self._token(rng, 4)}{rng.randint(10, 99)}",
            lambda: self._token(rng, rng.randint(20, 32)),
        ]
        return Simple(rng.choice(styles)(), "password")

    def api_key(self, rng: random.Random | None = None) -> Simple:
        rng = self._rng(rng)
        A = string.ascii_letters + string.digits
        styles = [
            lambda: "sk-test-" + self._token(rng, 24, A),
            lambda: "sk-proj-" + self._token(rng, 32, A),
            lambda: "AKIA" + self._token(rng, 16, string.ascii_uppercase + string.digits),
            lambda: "ghp_" + self._token(rng, 36, A),
            lambda: "xoxb-" + self._digits(rng, 11) + "-" + self._digits(rng, 12) + "-" + self._token(rng, 20, A),
            lambda: "AIza" + self._token(rng, 35, A + "-_"),
            lambda: "SG." + self._token(rng, 22, A + "-_") + "." + self._token(rng, 43, A + "-_"),
            lambda: "glpat-" + self._token(rng, 20, A + "-_"),
            lambda: "key-" + self._token(rng, 32, string.hexdigits.lower()),
            lambda: self._token(rng, 32, string.hexdigits.lower()),
        ]
        return Simple(rng.choice(styles)(), "api_key")

    def token(self, rng: random.Random | None = None) -> Simple:
        rng = self._rng(rng)
        B = string.ascii_letters + string.digits + "-_"
        styles = [
            lambda: f"eyJ{self._token(rng, rng.randint(14, 30), B)}.eyJ{self._token(rng, rng.randint(30, 80), B)}.{self._token(rng, rng.randint(32, 43), B)}",
            lambda: self._token(rng, rng.randint(32, 64), B),
            lambda: "ya29." + self._token(rng, 60, B),
            lambda: self._token(rng, 40, string.hexdigits.lower()),
            lambda: "rt_" + self._token(rng, 48, B),
        ]
        return Simple(rng.choice(styles)(), "token")

    def otp(self, rng: random.Random | None = None) -> Simple:
        rng = self._rng(rng)
        v = rng.choice([self._digits(rng, 6), self._digits(rng, 6), self._digits(rng, 4), f"{self._digits(rng, 3)}-{self._digits(rng, 3)}", self._token(rng, 8, string.ascii_uppercase + string.digits)])
        return Simple(v, "one_time_code")

    def connection_string(self, rng: random.Random | None = None) -> ConnString:
        rng = self._rng(rng)
        scheme = rng.choice(["postgres", "postgresql", "mysql", "mongodb", "redis", "sqlserver"])
        user = rng.choice(["app", "admin", "svc_batch", "toramon", "readonly", "db_user"])
        pw = self.password(rng).text
        host = rng.choice(["db.example.test", "db01.internal.example", "10.0.12.34", "localhost", "cache.example.invalid"])
        port = {"postgres": 5432, "postgresql": 5432, "mysql": 3306, "mongodb": 27017, "redis": 6379, "sqlserver": 1433}[scheme]
        db = rng.choice(["app", "orders", "crm", "main", "warehouse"])
        return ConnString(scheme, user, pw, host, port, db)

    # --- KEEP 系
    def company(self, rng: random.Random | None = None) -> Simple:
        rng = self._rng(rng)
        base = rng.choice(P.COMPANY_BASES)
        form, pos = rng.choice(P.COMPANY_FORMS)
        name = f"{form}{base}" if pos == "pre" else f"{base}{form}" if pos == "post" else base
        return Simple(name, "company", {"base": base}, emit_span=False)

    def department(self, rng: random.Random | None = None) -> str:
        return self._rng(rng).choice(P.DEPARTMENTS)

    def product_code(self, rng: random.Random | None = None) -> Simple:
        rng = self._rng(rng)
        pre = rng.choice(P.PRODUCT_PREFIXES)
        v = rng.choice([f"{pre}-{rng.randint(100, 9999)}", f"{pre}-{rng.randint(100, 999)}{rng.choice('ABKSX')}", f"{pre}{rng.randint(10, 99)}-{rng.randint(1000, 9999)}", f"PN {rng.randint(10, 99)}-{rng.randint(1000, 9999)}-{rng.randint(10, 99)}", f"{pre}{rng.randint(1000, 9999)}{rng.choice(['', '-J', '/W'])}"])
        return Simple(v, "product_code", emit_span=False)

    def version(self, rng: random.Random | None = None) -> Simple:
        rng = self._rng(rng)
        a, b, c = rng.randint(0, 12), rng.randint(0, 20), rng.randint(0, 40)
        v = rng.choice([f"v{a}.{b}.{c}", f"Ver.{a}.{b:02d}", f"{a}.{b}.{c}", f"{rng.randint(2023, 2027)}.{rng.randint(1, 12):02d}", f"{a}.{b}.{c}-rc{rng.randint(1, 3)}", f"バージョン{a}.{b}"])
        return Simple(v, "version", emit_span=False)

    def error_code(self, rng: random.Random | None = None) -> Simple:
        rng = self._rng(rng)
        pre = rng.choice(P.ERROR_PREFIXES)
        v = rng.choice([f"{pre}-{rng.randint(1000, 9999)}", f"{pre}_{rng.randint(100, 999)}", f"{pre}{rng.randint(10, 99)}", f"0x8007{rng.randint(0, 0xFFFF):04X}", f"HTTP {rng.choice([400, 401, 403, 404, 500, 502, 503, 504])}", f"E{rng.randint(100, 999)}"])
        return Simple(v, "error_code", emit_span=False)

    def request_id(self, rng: random.Random | None = None) -> Simple:
        """リクエスト ID／トレース ID／チケット番号（個人に紐づかない処理識別子 → KEEP）。UUID 形式を含む。"""
        rng = self._rng(rng)
        hexs = "0123456789abcdef"
        def uuid4():
            h = "".join(rng.choice(hexs) for _ in range(32))
            return f"{h[:8]}-{h[8:12]}-4{h[13:16]}-{rng.choice('89ab')}{h[17:20]}-{h[20:32]}"
        d = f"{rng.randint(2024, 2027)}{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}"
        v = rng.choice([uuid4(), f"req-{d}-{rng.randint(1, 9999):04d}", f"REQ-{rng.randint(100000, 999999)}", f"trace-{''.join(rng.choice(hexs) for _ in range(16))}",
                        f"TKT-{rng.randint(2024, 2027)}-{rng.randint(1, 99999):05d}", f"job_{''.join(rng.choice(hexs) for _ in range(12))}", f"x-request-id: {uuid4()}"])
        return Simple(v, "request_id", emit_span=False)

    def amount(self, rng: random.Random | None = None) -> Simple:
        rng = self._rng(rng)
        n = rng.choice([rng.randint(1, 99) * 100, rng.randint(1, 999) * 1000, rng.randint(1, 99) * 10000, rng.randint(100, 999999)])
        v = rng.choice([f"¥{n:,}", f"{n:,}円", f"￥{n:,}", fw_digits(f"{n:,}円"), f"{n}円", f"{n:,} JPY", f"税込{n:,}円", f"{n // 10000}万円" if n >= 10000 and n % 10000 == 0 else f"{n:,}円"])
        return Simple(v, "amount", emit_span=False)

    def invoice_no(self, rng: random.Random | None = None) -> Simple:
        rng = self._rng(rng)
        y = rng.randint(2024, 2027)
        v = rng.choice([f"INV-{y}-{rng.randint(1, 9999):04d}", f"{y}{rng.randint(1, 12):02d}{rng.randint(1, 9999):04d}", f"請-{y % 100}-{rng.randint(1, 999):03d}", f"IV{self._digits(rng, 8)}", f"T{self._digits(rng, 13)}"])
        return Simple(v, "invoice_no", emit_span=False)

    def place(self, rng: random.Random | None = None) -> Simple:
        return Simple(self._rng(rng).choice(P.PLACES), "place", emit_span=False)

    def public_person(self, rng: random.Random | None = None) -> Simple:
        return Simple(self._rng(rng).choice(N.PUBLIC_PERSONS), "public_person")

    # --- 統計
    def describe(self) -> dict:
        """KIND ごとの、この split が使える値（組み合わせ）の数。乱数生成の種類は到達可能な値域の概算。"""
        share = (SPLIT_RANGES[self.split][1] - SPLIT_RANGES[self.split][0]) / 100
        combos_ja = sum(1 for s in self._common_surnames + self._rare_surnames for g in self._given if self._ja_combo_ok(s, g))
        combos_foreign = sum(1 for s in N.FOREIGN_SURNAMES for g in N.FOREIGN_GIVEN if self._combo_ok(s[0], g[0]))
        combos_foreign_kanji = sum(1 for s in N.KANJI_FOREIGN_SURNAMES for g in N.KANJI_FOREIGN_GIVEN if self._combo_ok(s[0], g[0]))
        addr_combos = sum(1 for _, m in P.MUNICIPALITIES for t in P.TOWNS if self.in_split("address", m, t))
        persons = combos_ja + combos_foreign + combos_foreign_kanji
        approx = lambda n: int(n * share)  # noqa: E731
        d = {
            "_split": self.split,
            "_detail": {
                "surnames_common": len(self._common_surnames),
                "surnames_rare_in_split": len(self._rare_surnames),
                "surnames_rare_total": len(N.SURNAMES_RARE),
                "given_common": len(N.GIVEN_COMMON),
                "given_rare_in_split": len(self._rare_given),
                "given_rare_total": len(N.GIVEN_RARE),
                "foreign_given": len(N.FOREIGN_GIVEN),
                "foreign_surnames": len(N.FOREIGN_SURNAMES),
                "person_combos_ja": combos_ja,
                "person_combos_foreign_kana": combos_foreign,
                "person_combos_foreign_kanji": combos_foreign_kanji,
                "prefectures": len(P.PREFECTURES),
                "municipalities": len(P.MUNICIPALITIES),
                "towns": len(P.TOWNS),
                "address_combos": addr_combos,
                "email_domains": len(P.EMAIL_DOMAINS),
                "area_codes": len(AREA_CODES),
                "split_share": share,
            },
            "PERSON": persons, "PERSON_KANA": persons, "PERSON_ROMAJI": persons, "PERSON_FAMILY": len(self._common_surnames) + len(self._rare_surnames) + len(N.FOREIGN_SURNAMES) + len(N.KANJI_FOREIGN_SURNAMES),
            "PERSON_GIVEN": len(self._given) + len(N.FOREIGN_GIVEN) + len(N.KANJI_FOREIGN_GIVEN),
            "ADDRESS": addr_combos, "ADDRESS_PARTIAL": len(P.MUNICIPALITIES), "POSTAL": approx(900 * 10000),
            "EMAIL": persons * 12, "PHONE": approx(len(AREA_CODES) * 10**7), "PHONE_INTL": approx(len(AREA_CODES) * 10**7), "FAX": approx(21 * 10**7),
            "DATE_BIRTH": (2008 - 1938 + 1) * 365, "DATE_PERSONAL": (2027 - 2012 + 1) * 365,
            "ACCOUNT": approx(9 * 10**6), "MYNUMBER": approx(10**11), "LICENSE": approx(88 * 46 * 10**6 * 30), "PASSPORT": approx(8 * 10**7),
            "PENSION": approx(9 * 10**9), "INSURANCE": approx(4 * 10**7), "RESIDENCE_CARD": approx(26**4 * 10**8),
            "MEMBER_ID": approx(6 * 10**6), "CUSTOMER_ID": approx(5 * 10**6), "CONTRACT_ID": approx(4 * 10**4), "ORDER_ID": approx(6 * 10**4), "EMPLOYEE_ID": approx(4 * 10**5),
            "URL_PERSONAL": approx(8 * 62**10), "URL_TRACK": approx(4 * 36**6),
            "SECRET_PASSWORD": 10 * 10**4, "SECRET_APIKEY": 10 * 62**16, "SECRET_TOKEN": 5 * 62**32, "SECRET_OTP": 10**6, "SECRET_CONNSTR": 6 * 6 * 5 * 5 * 10**4,
            # KEEP
            "COMPANY": len(P.COMPANY_BASES) * len(P.COMPANY_FORMS), "COMPANY_ADDRESS": addr_combos, "COMPANY_PHONE": approx(3 * 10**6),
            "SHARED_EMAIL": 15 * len(P.COMPANY_DOMAINS), "PUBLIC_URL": 4 * len(P.COMPANY_DOMAINS) * 10 * 5, "DATE_BUSINESS": (2027 - 2023 + 1) * 365,
            "PRODUCT_CODE": len(P.PRODUCT_PREFIXES) * 5 * 9000, "VERSION": 6 * 13 * 21, "ERROR_CODE": len(P.ERROR_PREFIXES) * 6 * 900, "AMOUNT": 8 * 999, "INVOICE_NO": 5 * 9999,
            "CORP_NUMBER": 9 * 10**11, "PLACE": len(P.PLACES), "PUBLIC_PERSON": len(N.PUBLIC_PERSONS), "REQUEST_ID": approx(7 * 16**16),
        }
        return d
