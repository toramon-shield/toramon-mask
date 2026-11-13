#!/usr/bin/env python3
"""デモ用の上流モック（OpenAI 互換）。受け取った本文をそのまま echo する応答を返し、受信内容を標準出力に出す（ゲートウェイが何を外へ出したかを見るため）。
  .venv/bin/python scripts/mock_upstream.py --port 8799
"""
import argparse, json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); body = json.loads(self.rfile.read(n) or b"{}")
        msgs = body.get("messages") or []; inp = body.get("input")
        text = msgs[-1]["content"] if msgs else (inp if isinstance(inp, str) else json.dumps(inp, ensure_ascii=False))
        print("UPSTREAM RECEIVED:", text, flush=True)
        if self.path.endswith("/responses"):
            out = {"id": "resp_mock", "object": "response", "model": body.get("model"), "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": f"受領: {text}"}]}]}
        else:
            out = {"id": "chatcmpl_mock", "object": "chat.completion", "model": body.get("model"), "choices": [{"index": 0, "message": {"role": "assistant", "content": f"受領しました: {text}"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        b = json.dumps(out, ensure_ascii=False).encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        b = json.dumps({"object": "list", "data": [{"id": "mock-model"}]}).encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8799); a = ap.parse_args()
print(f"mock upstream on 127.0.0.1:{a.port}", flush=True); ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
