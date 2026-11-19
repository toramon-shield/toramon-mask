"""復元（restore_token）の仕様: 使い捨て・期限（上限 60 秒）・期限切れは伏せたまま・既定は無効・ループバック限定。"""
import importlib.util, json, sys, threading, urllib.error, urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from toramon.vault import RestoreVault, restore_text, restore_request, is_loopback  # noqa: E402
from toramon.spans import Span  # noqa: E402
from toramon.filter import ToramonFilter, Policy  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


gw = _load("gw_restore", "toramon_gateway.py")
cli = _load("cli_restore", "toramon_cli.py")


class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


class FakeDetector:
    """「山田太郎」と「090-1234-5678」だけを見つける検出器（Span を返す）。"""
    def detect(self, text):
        out = []
        for val, lab in (("山田太郎", "private_person"), ("090-1234-5678", "private_phone")):
            i = text.find(val)
            while i >= 0:
                out.append(Span(i, i + len(val), lab, val, source="fake")); i = text.find(val, i + 1)
        return out


def _filter():
    return ToramonFilter(FakeDetector(), "fake", policy=Policy(use_rules=False, use_boundary_fix=False, use_keep_suppression=False))


# ---- 保管庫 ---------------------------------------------------------------

def test_token_is_single_use_and_mapping_is_copied():
    v = RestoreVault(60, clock=Clock(), sweep_interval=0)
    src = {"[PERSON_1]": "山田太郎"}; tok, ttl = v.put(src); src.clear()
    assert ttl == 60 and len(tok) >= 40
    m = v.take(tok); assert m == {"[PERSON_1]": "山田太郎"}
    assert v.take(tok) is None and len(v) == 0


def test_expired_token_is_refused_and_swept():
    c = Clock(); v = RestoreVault(60, clock=c, sweep_interval=0)
    tok, _ = v.put({"[PERSON_1]": "山田太郎"}); tok2, _ = v.put({"[PERSON_1]": "x"})
    c.t = 60.0
    assert v.take(tok) is None
    v.sweep(); assert len(v) == 0 and v.take(tok2) is None


def test_ttl_above_60s_is_rejected():
    with pytest.raises(ValueError): RestoreVault(61, sweep_interval=0)
    with pytest.raises(ValueError): RestoreVault(0, sweep_interval=0)


def test_capacity_evicts_oldest():
    v = RestoreVault(60, max_entries=2, clock=Clock(), sweep_interval=0)
    t1, _ = v.put({"[A_1]": "a"}); t2, _ = v.put({"[A_1]": "b"}); t3, _ = v.put({"[A_1]": "c"})
    assert v.take(t1) is None and v.take(t2) and v.take(t3)


def test_restore_request_keeps_text_masked_when_expired():
    c = Clock(); v = RestoreVault(60, clock=c, sweep_interval=0)
    tok, _ = v.put({"[PERSON_1]": "山田太郎"})
    code, out = restore_request(v, {"restore_token": tok, "text": "[PERSON_1] 様 [PERSON_2]"})
    assert code == 200 and out["text"] == "山田太郎 様 [PERSON_2]" and out["replaced"] == 1
    code, out = restore_request(v, {"restore_token": tok, "text": "[PERSON_1] 様"})
    assert code == 410 and out["text"] == "[PERSON_1] 様" and out["restored"] is False
    tok, _ = v.put({"[PERSON_1]": "山田太郎"}); c.t = 61
    code, out = restore_request(v, {"restore_token": tok, "text": "[PERSON_1]"})
    assert code == 410 and out["text"] == "[PERSON_1]"
    code, out = restore_request(v, {"restore_token": "guess", "text": "[PERSON_1]"})
    assert code == 410


def test_discard():
    v = RestoreVault(60, clock=Clock(), sweep_interval=0)
    tok, _ = v.put({"[PERSON_1]": "山田太郎"})
    assert restore_request(v, {"restore_token": tok, "discard": True}) == (200, {"discarded": True})
    assert v.take(tok) is None


def test_is_loopback():
    assert is_loopback("127.0.0.1") and is_loopback("::1") and is_loopback("localhost")
    assert not is_loopback("0.0.0.0") and not is_loopback("192.168.1.5")


def test_redact_mapping_matches_placeholders():
    mp = {}; res = _filter().redact("山田太郎と山田太郎、090-1234-5678", mapping=mp)
    assert res.redacted_text == "[PERSON_1]と[PERSON_1]、[PHONE_1]"
    assert mp == {"[PERSON_1]": "山田太郎", "[PHONE_1]": "090-1234-5678"}
    assert restore_text(res.redacted_text, mp)[0] == "山田太郎と山田太郎、090-1234-5678"


# ---- ゲートウェイの /redact と /restore ---------------------------------------

class FakeFilt:
    model_version = rules_version = "fake"
    def detect(self, text):
        from toramon.filter import FilterResult
        spans = [{"start": s.start, "end": s.end, "label": s.label} for s in FakeDetector().detect(text)]
        return FilterResult(status="ok", spans=spans, redacted_text=None, elapsed_ms=0.1, model_version="fake", rules_version="fake", policy_version="fake")


@pytest.fixture
def gateway(tmp_path):
    servers = []

    def start(vault):
        g = gw.Gateway(FakeFilt(), "http://127.0.0.1:9", tmp_path / "audit.jsonl", set(), True, vault=vault)
        srv = ThreadingHTTPServer(("127.0.0.1", 0), gw.make_handler(g)); threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv); return f"http://127.0.0.1:{srv.server_address[1]}"
    yield start
    for s in servers: s.shutdown()


def _post(url, obj):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as r: return r.status, json.loads(r.read()), r.headers
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read()), e.headers


def test_gateway_restore_round_trip(gateway):
    c = Clock(); url = gateway(RestoreVault(60, clock=c, sweep_interval=0))
    code, out, h = _post(url + "/redact", {"text": "山田太郎 090-1234-5678", "restorable": True})
    assert code == 200 and out["redacted_text"] == "[PERSON_1] [PHONE_1]" and out["expires_in"] == 60
    assert "山田太郎" not in json.dumps(out, ensure_ascii=False) and h["Cache-Control"] == "no-store"
    code, out, _ = _post(url + "/restore", {"restore_token": out["restore_token"], "text": "宛先: [PERSON_1]（[PHONE_1]）"})
    assert code == 200 and out["text"] == "宛先: 山田太郎（090-1234-5678）"


def test_gateway_restore_expired_stays_masked(gateway):
    c = Clock(); url = gateway(RestoreVault(60, clock=c, sweep_interval=0))
    _, out, _ = _post(url + "/redact", {"text": "山田太郎", "restorable": True}); c.t = 60
    code, out, _ = _post(url + "/restore", {"restore_token": out["restore_token"], "text": "[PERSON_1]"})
    assert code == 410 and out["text"] == "[PERSON_1]"


def test_gateway_restore_disabled_by_default(gateway):
    url = gateway(None)
    code, out, _ = _post(url + "/redact", {"text": "山田太郎"})
    assert code == 200 and "restore_token" not in out
    assert _post(url + "/redact", {"text": "山田太郎", "restorable": True})[0] == 400
    assert _post(url + "/restore", {"restore_token": "x", "text": "[PERSON_1]"})[0] == 400


def test_gateway_cache_does_not_hold_raw_text():
    cf = gw.CachedFilter(FakeFilt()); cf.detect("山田太郎さん")
    assert all(isinstance(k, bytes) and "山田太郎".encode() not in k for k in cf.cache)


# ---- CLI の run ---------------------------------------------------------------

def test_cli_run_restores_command_output():
    rc, out, ok = cli.run_restoring(_filter(), "山田太郎に 090-1234-5678 で連絡", [sys.executable, "-c", "import sys; print('返信: ' + sys.stdin.read())"])
    assert rc == 0 and ok and out.strip() == "返信: 山田太郎に 090-1234-5678 で連絡"


def test_cli_run_sees_only_masked_input(tmp_path):
    f = tmp_path / "seen.txt"
    cli.run_restoring(_filter(), "山田太郎", [sys.executable, "-c", f"import sys; open({str(f)!r},'w').write(sys.stdin.read())"])
    assert f.read_text() == "[PERSON_1]"


def test_cli_run_expired_output_stays_masked():
    rc, out, ok = cli.run_restoring(_filter(), "山田太郎", [sys.executable, "-c", "import sys,time; t=sys.stdin.read(); time.sleep(0.5); print(t)"], ttl=0.1)
    assert not ok and out.strip() == "[PERSON_1]"
