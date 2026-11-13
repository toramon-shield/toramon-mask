"""合成値プールとプレースホルダ挿入（Toramon Mask データ生成）。"""
from .fill import KIND_SPECS, KEEP_KINDS, MASK_KINDS, FilledDoc, FillError, fill_document, kind_of_entity, to_skeleton
from .pools import (
    Address,
    DateVal,
    Email,
    IdValue,
    Person,
    Phone,
    Pool,
    Rendered,
    Simple,
    VariantError,
    bucket,
    corp_number_check_digit,
    is_valid_corp_number,
    is_valid_my_number,
    my_number_check_digit,
    split_of,
)

__all__ = [
    "Address", "DateVal", "Email", "IdValue", "Person", "Phone", "Pool", "Rendered", "Simple", "VariantError",
    "bucket", "split_of", "my_number_check_digit", "is_valid_my_number", "corp_number_check_digit", "is_valid_corp_number",
    "KIND_SPECS", "KEEP_KINDS", "MASK_KINDS", "FilledDoc", "FillError", "fill_document", "kind_of_entity", "to_skeleton",
]
