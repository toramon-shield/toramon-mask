#!/usr/bin/env python3
"""ローカルフィルタ CLI。標準入力のテキストを検出・置換して出力する（外部通信なし）。

  echo "..." | python scripts/toramon_cli.py redact --backend S --gguf <path>
  echo "..." | python scripts/toramon_cli.py detect --backend S --gguf <path> --json
  python scripts/toramon_cli.py serve --backend S --gguf <path> --port 8765   # ローカル API（FastAPI）
  echo "..." | python scripts/toramon_cli.py redact --backend monju --model-dir models/monju-mask-ja-01   # CPU 向けの小型モデル

伏せた値を元に戻す（任意。対応表はメモリだけに置き、1 回戻すか 60 秒経つかの早い方で消える）:
  cat memo.txt | python scripts/toramon_cli.py run --backend monju --model-dir models/monju-mask-ja-01 -- <LLM を呼ぶコマンド>
      … 伏せた文章をコマンドの標準入力へ渡し、コマンドの標準出力を元に戻して表示する。期限を過ぎたら伏せたまま表示して終了コード 3
  echo "..." | python scripts/toramon_cli.py redact --server http://127.0.0.1:8765 --restorable --json   # 起動中の API に対応表を預ける
  echo "[PERSON_1] さん" | python scripts/toramon_cli.py restore --server http://127.0.0.1:8765 --token <restore_token>

モデルの切り替え（--backend）: monju = Monju Mask（ONNX、CPU のみ）、S = sumeshi/privacy-filter-jp（GGUF）、
T-O / T-S = このリポジトリの手順で追加学習したモデル（GGUF か HF 形式）、O = openai/privacy-filter（公式ランタイム）。
"""
from __future__ import annotations
import argparse, json, subprocess, sys, threading, urllib.error, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toramon.filter import ToramonFilter, Policy  # noqa: E402
from toramon.vault import MAX_TTL, restore_text  # noqa: E402

EXIT_RESTORE_EXPIRED = 3


def build(a) -> ToramonFilter:
    if a.backend == "monju":
        from toramon.adapters.monju_onnx import load_monju
        d = a.model_dir or a.gguf
        if not d: raise SystemExit("--backend monju には --model-dir（monju.json と *.onnx を置いたフォルダ）が必要です")
        det = load_monju(d, threads=a.threads); mv = f"monju:{Path(d).name}"
    elif a.backend == "S":
        from toramon.adapters.sumeshi_pf import SumeshiPF
        det = SumeshiPF(a.gguf, device=a.device, threshold=a.threshold)
        mv = f"{a.backend}:{Path(a.gguf).name}"
    elif a.gguf:  # T-O / T-S の GGUF: 生ロジットを Toramon 復号器（HF adapter と同一）で復号する。libpf 既定の復号器では T-O のラベルが写像されない
        from toramon.adapters.sumeshi_pf import SumeshiPFDecoded
        det = SumeshiPFDecoded(a.gguf, device=a.device, threshold=a.threshold, label_space="identity" if a.backend == "T-O" else "S")
        mv = f"{a.backend}:{Path(a.gguf).name}:toramon-decode"
    else:
        from toramon.adapters.openai_opf import OpenAIOPF
        det = OpenAIOPF(a.checkpoint, device=a.device)
        mv = f"{a.backend}:{Path(a.checkpoint).name}"
    pol = Policy(use_rules=not a.no_rules, use_boundary_fix=not a.no_rules, use_keep_suppression=not a.no_rules, placeholder_style=a.style)
    return ToramonFilter(det, mv, policy=pol)


def run_restoring(filt: ToramonFilter, text: str, cmd: list[str], ttl: float = MAX_TTL, timeout: float | None = None) -> tuple[int, str, bool]:
    """text を伏せて cmd の標準入力へ渡し、cmd の標準出力を元に戻す。
    対応表はこの関数の中だけで持ち、ttl 秒経つか元に戻し終えたら消す。返り値は（終了コード, 出力, 元に戻せたか）。"""
    mapping: dict = {}; lock = threading.Lock(); expired = threading.Event()

    def wipe():
        with lock: mapping.clear(); expired.set()

    res = filt.redact(text, mapping=mapping)
    if res.status != "ok":
        mapping.clear(); raise RuntimeError(f"filter error: {res.error}")
    timer = threading.Timer(ttl, wipe); timer.daemon = True; timer.start()
    try:
        p = subprocess.run(cmd, input=res.redacted_text, stdout=subprocess.PIPE, text=True, timeout=timeout)
        with lock:
            if expired.is_set(): return p.returncode, p.stdout, False
            out, _ = restore_text(p.stdout, mapping)
            return p.returncode, out, True
    finally:
        timer.cancel(); wipe()


def post(url: str, obj: dict) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=json.dumps(obj, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r: return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["detect", "redact", "serve", "restore", "run"])
    ap.add_argument("--backend", default="S", choices=["monju", "S", "O", "T-O", "T-S"])
    ap.add_argument("--model-dir", help="backend monju: monju.json と *.onnx を置いたフォルダ（--gguf に同じパスを渡してもよい）"); ap.add_argument("--threads", type=int, default=4, help="backend monju: 使う CPU スレッド数")
    ap.add_argument("--gguf"); ap.add_argument("--checkpoint"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--threshold", type=float, default=0.5); ap.add_argument("--no-rules", action="store_true")
    ap.add_argument("--style", default="label", choices=["label", "asterisk"]); ap.add_argument("--json", action="store_true")
    ap.add_argument("--port", type=int, default=8765); ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--allow-restore", action="store_true", help="serve: /redact の restorable と /restore を有効にする（ループバックのときだけ）")
    ap.add_argument("--ttl", type=float, default=MAX_TTL, help="復元用の対応表を持つ秒数（上限 60）。serve と run で使う")
    ap.add_argument("--server", help="redact / restore: 起動中のローカル API かゲートウェイの URL（例 http://127.0.0.1:8765）")
    ap.add_argument("--restorable", action="store_true", help="redact --server: 対応表をサーバーに預け、使い捨ての restore_token を受け取る")
    ap.add_argument("--token", help="restore: redact --restorable で受け取った restore_token")
    argv = sys.argv[1:]; cmd = []
    if "--" in argv: i = argv.index("--"); argv, cmd = argv[:i], argv[i + 1:]   # run: -- の後は伏せた文章を標準入力で受け取るコマンド
    a = ap.parse_args(argv)
    if not 0 < a.ttl <= MAX_TTL: raise SystemExit("--ttl は 0 より大きく 60 以下にしてください")

    if a.cmd == "restore":
        if not (a.server and a.token): raise SystemExit("restore には --server と --token が必要です（CLI 単体は対応表を持たないため）")
        code, out = post(a.server.rstrip("/") + "/restore", {"restore_token": a.token, "text": sys.stdin.read()})
        if code == 200: sys.stdout.write(out["text"]); return 0
        if code == 410:
            sys.stdout.write(out["text"]); print("toramon: 復元の期限切れか使用済みのため、伏せたまま出力しました", file=sys.stderr); return EXIT_RESTORE_EXPIRED
        print(json.dumps(out, ensure_ascii=False), file=sys.stderr); return 1

    if a.cmd == "redact" and a.server:
        code, out = post(a.server.rstrip("/") + "/redact", {"text": sys.stdin.read(), "restorable": a.restorable})
        if code != 200 or out.get("status", "ok") != "ok": print(json.dumps(out, ensure_ascii=False), file=sys.stderr); return 1
        if a.json: print(json.dumps(out, ensure_ascii=False, indent=1))
        else:
            sys.stdout.write(out["redacted_text"])
            if "restore_token" in out: print(f"restore_token={out['restore_token']} expires_in={out['expires_in']:g}", file=sys.stderr)
        return 0
    if a.restorable:
        raise SystemExit("CLI 単体では対応表を残しません。--server で起動中の API に預けるか、run サブコマンドを使ってください")

    f = build(a)
    if a.cmd == "serve":
        from toramon.api import serve
        serve(f, a.host, a.port, allow_restore=a.allow_restore, restore_ttl=a.ttl); return 0
    if a.cmd == "run":
        if not cmd: raise SystemExit("run には -- の後にコマンドが必要です")
        rc, out, ok = run_restoring(f, sys.stdin.read(), cmd, a.ttl)
        sys.stdout.write(out)
        if not ok:
            print(f"toramon: コマンドが {a.ttl:g} 秒以内に終わらなかったため対応表を消し、伏せたまま出力しました", file=sys.stderr)
            return rc or EXIT_RESTORE_EXPIRED
        return rc
    text = sys.stdin.read()
    res = f.redact(text) if a.cmd == "redact" else f.detect(text)
    if a.json or a.cmd == "detect":
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=1))
    else:
        if res.status != "ok":
            print(json.dumps(res.to_dict(), ensure_ascii=False), file=sys.stderr); return 1
        sys.stdout.write(res.redacted_text)
    return 0 if res.status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
