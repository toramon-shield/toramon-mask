"""ゲートウェイの Anthropic 形式・Responses 形式の扱い（モデルなしで確かめられる部分）。"""
import importlib.util, json, sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("gw", Path(__file__).resolve().parents[1] / "scripts" / "toramon_gateway.py")
gw = importlib.util.module_from_spec(spec); sys.modules["gw"] = gw; spec.loader.exec_module(gw)


class FakeFilter:
    """「山田太郎」と「090-1234-5678」だけを見つける検出器。"""
    model_version = rules_version = "fake"

    def detect(self, text):
        from toramon.filter import FilterResult
        spans = []
        for val, lab in (("山田太郎", "private_person"), ("090-1234-5678", "private_phone")):
            i = text.find(val)
            while i >= 0:
                spans.append({"start": i, "end": i + len(val), "label": lab}); i = text.find(val, i + 1)
        return FilterResult(status="ok", spans=spans, redacted_text=None, elapsed_ms=0.1, model_version="fake", rules_version="fake", policy_version="fake")


def _masker():
    return gw.Masker(gw.CachedFilter(FakeFilter()))


def test_anthropic_request_masks_messages_but_not_system_or_thinking():
    m = _masker()
    obj = {"system": [{"type": "text", "text": "山田太郎"}], "tools": [{"name": "Bash", "description": "山田太郎"}],
           "messages": [{"role": "user", "content": "山田太郎さんに 090-1234-5678 で連絡"},
                        {"role": "assistant", "content": [{"type": "thinking", "thinking": "山田太郎", "signature": "sig"},
                                                          {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "echo 山田太郎"}}]},
                        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "山田太郎"}]}]}]}
    gw.mask_request("anthropic", obj, m, mask_system=False)
    s = json.dumps(obj, ensure_ascii=False)
    assert obj["messages"][0]["content"] == "[PERSON_1]さんに [PHONE_1] で連絡"
    assert obj["messages"][1]["content"][1]["input"]["command"] == "echo [PERSON_1]"
    assert obj["messages"][2]["content"][0]["content"][0]["text"] == "[PERSON_1]"
    assert obj["messages"][1]["content"][0]["thinking"] == "山田太郎"          # 署名つきの思考は触らない
    assert obj["system"][0]["text"] == "山田太郎" and obj["tools"][0]["description"] == "山田太郎"
    assert s.count("090-1234-5678") == 0


def test_responses_request_masks_input_and_arguments():
    m = _masker()
    obj = {"instructions": "山田太郎", "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "山田太郎"}]},
                                              {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": json.dumps({"cmd": "echo 山田太郎"}, ensure_ascii=False)},
                                              {"type": "function_call_output", "call_id": "c1", "output": "山田太郎"}]}
    gw.mask_request("responses", obj, m, mask_system=False)
    assert obj["input"][0]["content"][0]["text"] == "[PERSON_1]" and "山田太郎" not in obj["input"][1]["arguments"] and obj["input"][2]["output"] == "[PERSON_1]"
    assert obj["instructions"] == "山田太郎"


def test_stream_restore_joins_split_placeholders():
    m = _masker(); m.mask("山田太郎 090-1234-5678")
    sr = gw.StreamRestorer(m); out = []
    for piece in ["こんにちは [PER", "SON_1] さん、電話は [PHONE", "_1", "] です"]:
        for e in sr.event({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": piece}}): out.append(e["delta"]["text"])
    for e in sr.event({"type": "content_block_stop", "index": 0}):
        if e["type"] == "content_block_delta": out.append(e["delta"]["text"])
    assert "".join(out) == "こんにちは 山田太郎 さん、電話は 090-1234-5678 です"


def test_stream_restore_tool_json_is_escaped():
    m = _masker(); m.map["[PERSON_1]"] = '山田 "太郎"'
    sr = gw.StreamRestorer(m); parts = []
    for piece in ['{"command": "echo [PERS', 'ON_1]"}']:
        for e in sr.event({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": piece}}): parts.append(e["delta"]["partial_json"])
    for e in sr.event({"type": "content_block_stop", "index": 1}):
        if e["type"] == "content_block_delta": parts.append(e["delta"]["partial_json"])
    assert json.loads("".join(parts)) == {"command": 'echo 山田 "太郎"'}


def test_responses_stream_flushes_before_done():
    m = _masker(); m.mask("山田太郎")
    sr = gw.StreamRestorer(m); txt = []
    base = {"item_id": "i1", "output_index": 0, "content_index": 0}
    for piece in ["担当は [PERSON", "_1]"]:
        for e in sr.event({"type": "response.output_text.delta", **base, "delta": piece}): txt.append(e["delta"])
    done = sr.event({"type": "response.output_text.done", **base, "text": "担当は [PERSON_1]"})
    txt += [e["delta"] for e in done if e["type"].endswith(".delta")]
    assert "".join(txt) == "担当は 山田太郎" and done[-1]["text"] == "担当は 山田太郎"
