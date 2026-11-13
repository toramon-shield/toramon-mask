"""プレースホルダ文書への合成値挿入と位置確定（docs/data_design.md §1）。

テンプレート書式: ``{{<KIND>_<n>[:<variant>]}}``。同じ ``_n`` は同一実体。
- PERSON_1 / PERSON_KANA_1 / PERSON_FAMILY_1 / PERSON_ROMAJI_1 は同一人物（entity_id 共有）。
- EMAIL_1 / URL_PERSONAL_1 / SECRET_PASSWORD_1 は PERSON_1 が文書内にあればそのローマ字から派生する
  （entity_id は種類ごとに別: email-001 など）。
- ADDRESS_1 / ADDRESS_PARTIAL_1 / POSTAL_1 は同一住所。PHONE_1 / PHONE_INTL_1 は同一番号。
- variant は値クラスの描画表（pools.py）に従う。``+`` で生成フラグを連結できる（例 ``MYNUMBER_1:invalid+spaced4``）。

出力 entity は configs/data_schema.json の entities 要素（start, end, text, label, action, entity_id, subtype, source）。
位置は Unicode コードポイント単位・0 始まり・終端含まず。KEEP 系で label を持たない種類（COMPANY, PRODUCT_CODE,
VERSION, ERROR_CODE, AMOUNT, INVOICE_NO, PLACE）は entity を出さず FilledDoc.keep_values に記録する。
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any

from ..spans import Span, validate_spans
from . import pools as PL

PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Z][A-Z_]*?[A-Z])(?:_(\d+))?(?::([A-Za-z0-9_+\-]+))?\s*\}\}")
LEFTOVER_RE = re.compile(r"\{\{[^{}]*\}\}")


class FillError(ValueError):
    pass


@dataclass(frozen=True)
class KindSpec:
    kind: str
    group: str
    label: str | None
    action: str
    maker: str  # Pool のメソッド名
    person_linked: bool = False  # 同じ n の PERSON から派生する


_K = KindSpec
KIND_SPECS: dict[str, KindSpec] = {s.kind: s for s in (
    # MASK
    _K("PERSON", "person", "private_person", "MASK", "person"),
    _K("PERSON_KANA", "person", "private_person", "MASK", "person"),
    _K("PERSON_ROMAJI", "person", "private_person", "MASK", "person"),
    _K("PERSON_FAMILY", "person", "private_person", "MASK", "person"),
    _K("PERSON_GIVEN", "person", "private_person", "MASK", "person"),
    _K("ADDRESS", "address", "private_address", "MASK", "address"),
    _K("ADDRESS_PARTIAL", "address", "private_address", "MASK", "address"),
    _K("POSTAL", "address", "private_address", "MASK", "address"),
    _K("EMAIL", "email", "private_email", "MASK", "email", True),
    _K("PHONE", "phone", "private_phone", "MASK", "phone"),
    _K("PHONE_INTL", "phone", "private_phone", "MASK", "phone"),
    _K("FAX", "fax", "private_phone", "MASK", "fax"),
    _K("DATE_BIRTH", "date_birth", "private_date", "MASK", "date_birth"),
    _K("DATE_PERSONAL", "date_personal", "private_date", "MASK", "date_personal"),
    _K("ACCOUNT", "account", "account_number", "MASK", "bank_account"),
    _K("MYNUMBER", "mynumber", "account_number", "MASK", "my_number"),
    _K("LICENSE", "license", "account_number", "MASK", "license_number"),
    _K("PASSPORT", "passport", "account_number", "MASK", "passport_number"),
    _K("PENSION", "pension", "account_number", "MASK", "pension_number"),
    _K("INSURANCE", "insurance", "account_number", "MASK", "insurance_number"),
    _K("RESIDENCE_CARD", "residence_card", "account_number", "MASK", "residence_card"),
    _K("MEMBER_ID", "member_id", "account_number", "MASK", "member_id"),
    _K("CUSTOMER_ID", "customer_id", "account_number", "MASK", "customer_id"),
    _K("CONTRACT_ID", "contract_id", "account_number", "MASK", "contract_id"),
    _K("ORDER_ID", "order_id", "account_number", "MASK", "order_id"),
    _K("EMPLOYEE_ID", "employee_id", "account_number", "MASK", "employee_id"),
    _K("URL_PERSONAL", "url_personal", "private_url", "MASK", "url_personal", True),
    _K("URL_TRACK", "url_track", "private_url", "MASK", "url_track"),
    _K("SECRET_PASSWORD", "secret_password", "secret", "MASK", "password", True),
    _K("SECRET_APIKEY", "secret_apikey", "secret", "MASK", "api_key"),
    _K("SECRET_TOKEN", "secret_token", "secret", "MASK", "token"),
    _K("SECRET_OTP", "secret_otp", "secret", "MASK", "otp"),
    _K("SECRET_CONNSTR", "secret_connstr", "secret", "MASK", "connection_string"),
    # KEEP（label あり → action=KEEP の entity を出す）
    _K("COMPANY_ADDRESS", "company_address", "private_address", "KEEP", "company_address"),
    _K("COMPANY_PHONE", "company_phone", "private_phone", "KEEP", "company_phone"),
    _K("SHARED_EMAIL", "shared_email", "private_email", "KEEP", "shared_email"),
    _K("PUBLIC_URL", "public_url", "private_url", "KEEP", "public_url"),
    _K("DATE_BUSINESS", "date_business", "private_date", "KEEP", "date_business"),
    _K("CORP_NUMBER", "corp_number", "account_number", "KEEP", "corp_number"),
    _K("PUBLIC_PERSON", "public_person", "private_person", "KEEP", "public_person"),
    # KEEP（label なし → entity を出さず keep_values に記録）
    _K("COMPANY", "company", None, "KEEP", "company"),
    _K("PRODUCT_CODE", "product_code", None, "KEEP", "product_code"),
    _K("VERSION", "version", None, "KEEP", "version"),
    _K("ERROR_CODE", "error_code", None, "KEEP", "error_code"),
    _K("AMOUNT", "amount", None, "KEEP", "amount"),
    _K("INVOICE_NO", "invoice_no", None, "KEEP", "invoice_no"),
    _K("PLACE", "place", None, "KEEP", "place"),
    _K("REQUEST_ID", "request_id", None, "KEEP", "request_id"),
)}

MASK_KINDS: tuple[str, ...] = tuple(k for k, s in KIND_SPECS.items() if s.action == "MASK")
KEEP_KINDS: tuple[str, ...] = tuple(k for k, s in KIND_SPECS.items() if s.action == "KEEP")

# 生成時フラグ（variant に ``+`` で連結）
GENERATION_FLAGS = {"invalid", "valid", "foreign", "rare", "male", "female", "mobile", "landline", "zerowidth", "zero_width"}


@dataclass
class FilledDoc:
    text: str
    entities: list[dict] = field(default_factory=list)
    keep_values: list[dict] = field(default_factory=list)
    fills: list[dict] = field(default_factory=list)  # 全プレースホルダの挿入記録
    values: dict[str, Any] = field(default_factory=dict)  # entity_id → 値オブジェクト

    def skeleton(self) -> str:
        return to_skeleton(self.text, self.entities, self.keep_values)

    def spans(self) -> list[Span]:
        return [Span(e["start"], e["end"], e["label"], e["text"], e["action"], e["entity_id"], source=e.get("source", "")) for e in self.entities]


@dataclass
class _Ref:
    name: str  # "PERSON_1"
    kind: str
    index: int
    variant: str | None
    flags: set[str]
    spec: KindSpec | None
    override: dict


def _parse_variant(raw: str | None) -> tuple[str | None, set[str]]:
    if not raw:
        return None, set()
    toks = [t for t in raw.split("+") if t]
    flags = {t for t in toks if t in GENERATION_FLAGS}
    rest = [t for t in toks if t not in GENERATION_FLAGS]
    if len(rest) > 1:
        raise FillError(f"at most one render variant allowed, got {rest}")
    return (rest[0] if rest else None), flags



def _insert_zero_width(rendered):
    """値の 2 文字目の前にゼロ幅スペース(U+200B)を挿入する（難易度軸 zero_width）。span はゼロ幅文字を含む（annotation_policy §3）。"""
    text = rendered.text
    if len(text) < 2:
        return rendered
    # 最初の part の内側（先頭文字の直後）に挿入し、以降の part のオフセットをずらす
    parts = list(rendered.parts)
    if not parts:
        return rendered
    p0 = parts[0]
    ins = p0.offset + 1
    if ins >= p0.end:
        return rendered
    new_text = text[:ins] + "\u200b" + text[ins:]
    new_parts = []
    for pt in parts:
        off = pt.offset + (1 if pt.offset >= ins else 0)
        length = pt.length + (1 if pt is p0 else 0)
        new_parts.append(type(pt)(offset=off, length=length, role=pt.role, subtype=pt.subtype))
    return type(rendered)(new_text, tuple(new_parts), dict(rendered.meta))


def _entity_id(group: str, index: int) -> str:
    return f"{group.replace('_', '-')}-{index:03d}"


def _make_value(pool: PL.Pool, rng: random.Random, spec: KindSpec, index: int, flags: set[str], persons: dict[int, PL.Person]) -> Any:
    m = spec.maker
    if m == "person":
        kw: dict[str, Any] = {}
        if "foreign" in flags:
            kw["foreign"] = True
        if flags & {"joint", "joint_space", "furigana_hira", "hiragana"}:
            kw["foreign"] = False  # 連名・ひらがな表記は日本人名でのみ自然
        if flags & {"kyujitai", "kyujitai_space"}:
            kw["kyujitai"] = True
        if "rare" in flags:
            kw["rare"] = True
        if "male" in flags:
            kw["gender"] = "m"
        if "female" in flags:
            kw["gender"] = "f"
        return pool.person(rng, **kw)
    if m == "my_number":
        return pool.my_number(rng, valid="invalid" not in flags)
    if m == "phone":
        kind = "mobile" if "mobile" in flags else "landline" if "landline" in flags else None
        return pool.phone(rng, kind=kind)
    if spec.person_linked:
        return getattr(pool, m)(rng, person=persons.get(index))
    return getattr(pool, m)(rng)


def _lookup_override(placeholders: dict | None, ref_name: str, kind: str, index: int) -> dict:
    if not placeholders:
        return {}
    for key in (ref_name, f"{{{{{ref_name}}}}}", f"{kind}_{index}"):
        if key in placeholders and isinstance(placeholders[key], dict):
            return placeholders[key]
    return {}


def fill_document(template_text: str, placeholders: dict | None, pool: PL.Pool, rng: random.Random | None = None) -> FilledDoc:
    """プレースホルダ文書に合成値を挿入し、コードポイント位置付きの entity を返す。

    placeholders: LLM が返した補助情報（任意）。キーは "PERSON_1" 形式。値は dict で
      label / action / entity_id / subtype / variant / value / note を上書きできる。
      未知の KIND は value が与えられたときのみ受け付ける（label があれば entity、なければ keep_values）。
    """
    rng = rng if rng is not None else pool.rng

    # --- 1. 走査
    refs: list[tuple[re.Match, _Ref]] = []
    for m in PLACEHOLDER_RE.finditer(template_text):
        kind = m.group(1)
        index = int(m.group(2)) if m.group(2) else 1
        name = f"{kind}_{index}"
        override = _lookup_override(placeholders, name, kind, index)
        variant, flags = _parse_variant(override.get("variant") or m.group(3))
        spec = KIND_SPECS.get(kind)
        if spec is None and "value" not in override:
            raise FillError(f"unknown placeholder kind {kind!r} in {m.group(0)!r} (no value supplied in placeholders)")
        refs.append((m, _Ref(name, kind, index, variant, flags, spec, override)))

    # --- 2. 実体の確定（PERSON を先に作り、派生種類はそれを参照）
    values: dict[tuple[str, int], Any] = {}
    flags_by_key: dict[tuple[str, int], set[str]] = {}
    for _, r in refs:
        if r.spec:
            fl = flags_by_key.setdefault((r.spec.group, r.index), set())
            fl.update(r.flags)
            if r.variant:
                fl.add(r.variant)  # 描画 variant も生成時のヒントにする（joint / kyujitai など）
    ordered = sorted({(r.spec.group, r.index, r.spec.maker) for _, r in refs if r.spec}, key=lambda k: (k[2] != "person", k[0], k[1]))
    persons: dict[int, PL.Person] = {}
    for group, index, maker in ordered:
        spec = next(s for s in KIND_SPECS.values() if s.group == group)
        val = _make_value(pool, rng, spec, index, flags_by_key.get((group, index), set()), persons)
        values[(group, index)] = val
        if maker == "person":
            persons[index] = val

    # --- 3. 描画と位置確定
    out: list[str] = []
    pos = 0
    cursor = 0
    doc = FilledDoc(text="")
    for m, r in refs:
        literal = template_text[cursor : m.start()]
        out.append(literal)
        pos += len(literal)
        cursor = m.end()

        ov = r.override
        if "value" in ov:
            val_text = str(ov["value"])
            label = ov.get("label") or (r.spec.label if r.spec else None)
            rendered = PL.Rendered.whole(val_text, ov.get("subtype")) if label else PL.Rendered.none(val_text)
            value_obj: Any = val_text
            base_eid = ov.get("entity_id") or _entity_id(r.spec.group if r.spec else r.kind.lower(), r.index)
            action = ov.get("action") or (r.spec.action if r.spec else "KEEP")
        else:
            assert r.spec is not None
            value_obj = values[(r.spec.group, r.index)]
            try:
                rendered = value_obj.render(r.kind, r.variant)
            except PL.VariantError as e:
                raise FillError(f"{m.group(0)}: {e}") from e
            if "zerowidth" in r.flags or "zero_width" in r.flags:
                rendered = _insert_zero_width(rendered)
            label = ov.get("label", r.spec.label)
            action = ov.get("action", r.spec.action)
            base_eid = ov.get("entity_id") or _entity_id(r.spec.group, r.index)

        start = pos
        out.append(rendered.text)
        pos += len(rendered.text)
        end = pos
        entity_ids: list[str] = []
        if label:
            for part in rendered.parts:
                eid = f"{base_eid}-joint" if part.role == "joint" else base_eid
                ent = {
                    "start": start + part.offset,
                    "end": start + part.end,
                    "text": rendered.text[part.offset : part.end],
                    "label": label,
                    "action": action,
                    "entity_id": eid,
                    "source": "placeholder",
                }
                subtype = ov.get("subtype") or part.subtype
                if subtype:
                    ent["subtype"] = subtype
                if ov.get("note"):
                    ent["note"] = str(ov["note"])
                doc.entities.append(ent)
                entity_ids.append(eid)
                doc.values.setdefault(eid, value_obj)
        else:
            doc.keep_values.append({"name": r.name, "kind": r.kind, "value": rendered.text, "start": start, "end": end})
        doc.fills.append({
            "name": r.name, "kind": r.kind, "index": r.index, "variant": r.variant, "flags": sorted(r.flags),
            "start": start, "end": end, "text": rendered.text, "entity_ids": entity_ids, "meta": dict(rendered.meta),
        })
    out.append(template_text[cursor:])
    doc.text = "".join(out)

    # --- 4. 検証
    left = LEFTOVER_RE.search(doc.text)
    if left:
        raise FillError(f"unfilled placeholder remains: {left.group(0)!r}")
    doc.entities.sort(key=lambda e: (e["start"], e["end"]))
    problems = validate_spans(doc.text, doc.spans())
    if problems:
        raise FillError("span validation failed: " + "; ".join(problems))
    return doc


# ---------------------------------------------------------------------------
# 骨格化（重複監査用）
# ---------------------------------------------------------------------------
_ACCOUNT_SUBTYPES = {
    "my_number": "MYNUMBER", "drivers_license": "LICENSE", "passport": "PASSPORT", "pension": "PENSION",
    "health_insurance": "INSURANCE", "bank_account": "ACCOUNT", "residence_card": "RESIDENCE_CARD",
    "member_id": "MEMBER_ID", "customer_id": "CUSTOMER_ID", "contract_id": "CONTRACT_ID", "order_id": "ORDER_ID",
    "employee_id": "EMPLOYEE_ID", "corp_number": "CORP_NUMBER",
}


def kind_of_entity(entity: dict) -> str:
    """entity の label/subtype/action から {{KIND}} 名を決める（骨格化用）。"""
    label, sub, keep = entity["label"], entity.get("subtype"), entity.get("action") == "KEEP"
    if label == "private_person":
        if keep:
            return "PUBLIC_PERSON"
        return {"furigana": "PERSON_KANA", "romaji": "PERSON_ROMAJI", "family_name": "PERSON_FAMILY", "given_name": "PERSON_GIVEN"}.get(sub or "", "PERSON")
    if label == "private_address":
        if keep:
            return "COMPANY_ADDRESS"
        return {"postal_code_only": "POSTAL", "partial": "ADDRESS_PARTIAL"}.get(sub or "", "ADDRESS")
    if label == "private_email":
        return "SHARED_EMAIL" if keep else "EMAIL"
    if label == "private_phone":
        if keep:
            return "COMPANY_PHONE"
        return {"fax": "FAX", "intl": "PHONE_INTL"}.get(sub or "", "PHONE")
    if label == "private_date":
        if keep:
            return "DATE_BUSINESS"
        return "DATE_BIRTH" if sub == "birth" else "DATE_PERSONAL"
    if label == "account_number":
        if keep:
            return "CORP_NUMBER"
        return _ACCOUNT_SUBTYPES.get(sub or "", "ACCOUNT")
    if label == "private_url":
        if keep:
            return "PUBLIC_URL"
        return "URL_TRACK" if sub == "tracking" else "URL_PERSONAL"
    if label == "secret":
        return {"password": "SECRET_PASSWORD", "api_key": "SECRET_APIKEY", "one_time_code": "SECRET_OTP", "connection_string": "SECRET_CONNSTR"}.get(sub or "", "SECRET_TOKEN")
    return label.upper()


def to_skeleton(text: str, entities: list[dict], keep_values: list[dict] | None = None) -> str:
    """entity（と任意で keep_values）のテキストを ``{{KIND}}`` に戻した骨格を返す。

    分割監査（docs/data_design.md §2「固有値を {{KIND}} に戻した骨格一致」）用。番号は付けない。
    """
    repl: list[tuple[int, int, str]] = [(e["start"], e["end"], kind_of_entity(e)) for e in entities]
    for k in keep_values or ():
        repl.append((k["start"], k["end"], k["kind"]))
    repl.sort(key=lambda t: (t[0], -t[1]))
    pieces: list[str] = []
    cursor = 0
    for s, e, kind in repl:
        if s < cursor:
            continue  # 重なり（想定外）は先勝ち
        if text[s:e] == "":
            continue
        pieces.append(text[cursor:s])
        pieces.append("{{" + kind + "}}")
        cursor = e
    pieces.append(text[cursor:])
    return "".join(pieces)
