"""ランタイム依存の統合テスト（libpf.so と GGUF がある場合のみ実行）。

CPU 版 libpf（PF_CUDA=OFF でビルド）で実行するときは TORAMON_TEST_DEVICE=cpu を指定する。
"""
import os, sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toramon.paths import SUMESHI_F16_GGUF, PFJP_DIR  # noqa: E402
G = str(SUMESHI_F16_GGUF)
LIB = PFJP_DIR / "build/cuda-shared/libpf.so"
pytestmark = pytest.mark.skipif(not (os.path.exists(G) and LIB.exists()), reason="libpf/GGUF not available")


@pytest.fixture(scope="module")
def model():
    from toramon.adapters.sumeshi_pf import SumeshiPF
    return SumeshiPF(G, device=os.environ.get("TORAMON_TEST_DEVICE", "cuda"))


def test_offsets_are_codepoints_and_texts_match(model):
    t = "👨‍👩‍👧 担当：髙橋　花子（タカハシ ハナコ）TEL ０９０－１２３４－５６７８ mail: hanako@example.jp"
    spans = model.detect(t)
    assert spans
    for s in spans:
        assert t[s.start:s.end] == s.text
    labels = {s.label for s in spans}
    assert "private_person" in labels and "private_phone" in labels and "private_email" in labels


def test_concurrent_calls_do_not_crash(model):
    from concurrent.futures import ThreadPoolExecutor
    texts = [f"山田太郎{i}さんの電話は090-1234-{i:04d}です。" for i in range(16)]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(model.detect, texts))
    assert all(len(r) >= 2 for r in res)


def test_empty_and_whitespace(model):
    assert model.detect("") == []
    assert model.detect("   \n\n") == []


def test_filter_error_paths(model):
    from toramon.filter import ToramonFilter, Policy
    f = ToramonFilter(model, "S-test", policy=Policy(max_chars=50))
    assert f.detect("あ" * 51).status == "error"
    assert f.detect("壊れた�文字").status == "error"
    r = f.redact("山田太郎様 TEL:090-1234-5678")
    assert r.status == "ok" and "090" not in r.redacted_text and "山田" not in r.redacted_text
