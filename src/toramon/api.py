"""ローカル HTTP API（FastAPI）。CLI と同じ ToramonFilter を呼ぶ。127.0.0.1 に bind し、外部通信は行わない。"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel


class TextIn(BaseModel):
    text: str
    style: str | None = None
    use_rules: bool | None = None


def make_app(filt) -> FastAPI:
    app = FastAPI(title="Toramon Mask local filter", version="0.1")

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
        return {"status": "ok", "model_version": filt.model_version, "rules_version": filt.rules_version}

    @app.post("/detect")
    def detect(inp: TextIn):
        return filt.detect(inp.text, _policy(inp)).to_dict()

    @app.post("/redact")
    def redact(inp: TextIn):
        return filt.redact(inp.text, _policy(inp)).to_dict()

    return app


def serve(filt, host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn
    uvicorn.run(make_app(filt), host=host, port=port, log_level="warning")
