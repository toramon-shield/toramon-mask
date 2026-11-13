"""共通ルール R（正規表現・形式検証・境界補正・辞書）。各ルールは ID・目的・対象カテゴリ・適用条件・優先順位・正例負例を持つ。"""
from .engine import RuleEngine, load_rules  # noqa: F401
