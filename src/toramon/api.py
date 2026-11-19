"""ローカル HTTP API（FastAPI）。CLI と同じ ToramonFilter を呼ぶ。127.0.0.1 に bind し、外部通信は行わない。

復元（任意、serve --allow-restore で有効）:
  POST /redact  {"text": ..., "restorable": true}      -> 伏せた文章と使い捨ての restore_token（対応表そのものは返さない）
  POST /restore {"text": 伏せた文章, "restore_token": ...} -> 元に戻した文章。鍵は 1 回で消える。期限（既定 60 秒）でも消える
  POST /restore {"restore_token": ..., "discard": true}   -> 戻さずに消すだけ
期限切れ・使用済みの鍵では 410 と伏せたままの文章を返す。
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

NO_STORE = {"Cache-Control": "no-store"}


class TextIn(BaseModel):
    text: str
    style: str | None = None
    use_rules: bool | None = None
    restorable: bool = False


class RestoreIn(BaseModel):
    restore_token: str | None = None
    text: str = ""
    discard: bool = False


def make_app(filt, vault=None) -> FastAPI:
    app = FastAPI(title="Toramon Mask local filter", version="0.2")

    def _policy(inp: TextIn):
        from .filter import Policy
        p = filt.policy
        return Policy(mask_labels=p.mask_labels,
                      use_rules=p.use_rules if inp.use_rules is None else inp.use_rules,
                      use_boundary_fix=p.use_boundary_fix if inp.use_rules is None else inp.use_rules,
                      use_keep_suppression=p.use_keep_suppression if inp.use_rules is None else inp.use_rules,
                      threshold=p.threshold, placeholder_style=inp.style or p.placeholder_style,
                      max_chars=p.max_chars, timeout_s=p.timeout_s, version=p.version)

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": filt.model_version, "rules_version": filt.rules_version, "restore": bool(vault)}

    @app.post("/detect")
    def detect(inp: TextIn):
        return filt.detect(inp.text, _policy(inp)).to_dict()

    @app.post("/redact")
    def redact(inp: TextIn):
        pol = _policy(inp)
        if not inp.restorable:
            return filt.redact(inp.text, pol).to_dict()
        if vault is None:
            return JSONResponse({"error": "restore is disabled; start the server with --allow-restore"}, 400)
        if pol.placeholder_style != "label":
            return JSONResponse({"error": "restorable requires style=label"}, 400)
        mapping: dict = {}
        try:
            res = filt.redact(inp.text, pol, mapping=mapping)
            out = res.to_dict()
            if res.status == "ok":
                out["restore_token"], out["expires_in"] = vault.put(mapping)
            return JSONResponse(out, headers=NO_STORE)
        finally:
            mapping.clear()

    @app.post("/restore")
    def restore(inp: RestoreIn):
        if vault is None:
            return JSONResponse({"error": "restore is disabled; start the server with --allow-restore"}, 400)
        from .vault import restore_request
        code, body = restore_request(vault, inp.model_dump())
        return JSONResponse(body, code, headers=NO_STORE)

    return app


def serve(filt, host: str = "127.0.0.1", port: int = 8765, allow_restore: bool = False, restore_ttl: float = 60.0) -> None:
    import uvicorn
    vault = None
    if allow_restore:
        from .vault import RestoreVault, is_loopback
        if not is_loopback(host):
            raise SystemExit("--allow-restore はループバック（127.0.0.1 / ::1 / localhost）で待ち受けるときだけ使えます（元の値がネットワークに流れるため）")
        vault = RestoreVault(restore_ttl)
    uvicorn.run(make_app(filt, vault), host=host, port=port, log_level="warning")
