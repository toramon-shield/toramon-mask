"""OpenAI privacy-filter の公式ランタイム（opf）アダプタ。コンテナ内で動かす。

- 公式の制約付き Viterbi デコーダ + viterbi_calibration.json を使う（decode_mode="viterbi"）。argmax は補助。
- opf の DetectedSpan はコードポイント位置（Python 文字列インデックス）。
- 不可分ではないが既定 ON の後処理: 空白トリム（trim_whitespace）。既定 ON で評価し、設定を記録する。
- ラベルは共通 8 カテゴリと同名（恒等写像）。カスタムラベル空間で学習した T-O も同じ経路で動く。
"""
from __future__ import annotations

import time

from ..spans import Span


class OpenAIOPF:
    def __init__(self, checkpoint: str, device: str = "cuda", decode_mode: str = "viterbi",
                 trim_whitespace: bool = True, context_window_length: int | None = None,
                 viterbi_calibration_path: str | None = None):
        from opf import OPF  # type: ignore
        from opf._api import DecodeOptions  # type: ignore
        t0 = time.perf_counter()
        self.opf = OPF(model=checkpoint, device=device, decode_mode=decode_mode, trim_whitespace=trim_whitespace,
                       context_window_length=context_window_length, output_mode="typed")
        self._decode = DecodeOptions(viterbi_calibration_path=viterbi_calibration_path) if viterbi_calibration_path else None
        # 初回ロードを済ませる
        self.opf.redact("warmup")
        self.load_seconds = time.perf_counter() - t0
        self.checkpoint = checkpoint
        self.decode_mode = decode_mode

    def detect(self, text: str) -> list[Span]:
        res = self.opf.redact(text, decode=self._decode) if self._decode else self.opf.redact(text)
        if res.text != text:
            # tokenizer round-trip 不一致（不正 UTF-8 等）。安全側: 例外にして呼び出し側で失敗扱いにする
            raise ValueError("decoded text mismatch; spans would not align with the original string")
        out = []
        for s in res.detected_spans:
            out.append(Span(s.start, s.end, s.label, text[s.start:s.end], "MASK", "", None, f"O:{self.decode_mode}"))
        out.sort(key=lambda s: (s.start, s.end))
        return out
