# Toramon Gateway（試作 v0.1、2026-11-11）

社員の PC またはお客様の環境で動く OpenAI 互換のローカルプロキシ。クラウド AI へ送る文章の個人情報を端末内で伏せ、応答では元の値に戻す。監査ログには件数だけを残し、原文は保存しない。

## 起動
```bash
.venv/bin/python scripts/toramon_gateway.py --backend S --gguf models/sumeshi-privacy-filter-jp-GGUF/privacy-filter-jp-f16.gguf --port 8787 --upstream https://api.openai.com
# 利用側（CLI / SDK / 社内ツール）: ベース URL をゲートウェイに向けるだけ。API キーは利用者のものをそのまま転送し、ゲートウェイは保存しない
export OPENAI_BASE_URL=http://127.0.0.1:8787/v1
```

## 機能
- `POST /v1/chat/completions`、`/v1/responses`、`/v1/embeddings` など `/v1/*` を上流へ転送。本文中の文章（content / input / text / prompt / instructions）を伏せてから送る。
- 応答の伏せ字（[PERSON_1] など）は、非ストリーミング応答では元の値に戻す。ストリーミング応答はチャンク単位で戻す（チャンク境界をまたぐ伏せ字は戻らないことがある、試作の既知制限）。
- `--block-labels secret`（既定）: パスワード・API キーを含む要求を 403 で止める。`--block-labels secret,account_number` のように拡張できる。
- `GET /audit/summary?days=7`: 件数・ラベル別・日別・モデル別・上流別の集計。`--audit` の JSONL が一次記録（1 要求 1 行、原文なし）。
- `POST /redact`: ローカルの伏せ字 API（他ツール連携用）。
- 共通ルール R v0.4 併用（企業の代表番号・共用メール・法人名などは残す）。

## 管理画面と利用者別集計
- `GET /admin`: 期間別の要求数・遮断数・伏せた箇所、種類別、日別、利用者別、送信先・モデル別、最近の要求（原文なし）を表示する管理画面（ゲートウェイ内蔵、外部ライブラリなし）。
- `GET /audit/by_user?days=7`、`GET /audit/recent?n=50`: 管理画面が使う集計 API。
- 利用者の帰属は `--attribute-user os|header|ip`（端末配布なら OS ユーザー名、共用サーバーなら接続元 IP か `X-Toramon-User` ヘッダ）。

## 周辺部品（試作、ブラウザ・Windows/macOS 上は未検証）
- `gateway/extension/`: Chrome / Edge 拡張（MV3）。ChatGPT・Claude・Gemini・Copilot の Web 画面で、貼り付けと Enter 送信の直前に `/redact` で伏せる。接続できないときは止める（fail-closed）。
- `gateway/desktop/toramon_clip.py`: クリップボード常駐。Copilot アプリ・Claude Desktop・ChatGPT Desktop など URL を変えられないアプリ向けに、コピーした文章を貼り付け前に伏せる。「元に戻す」ボタン付き。
- `gateway/packaging/`: PyInstaller での実行形式化（Linux/macOS 用 sh、Windows 用 ps1）と MDM 配布の雛形（Intune / Jamf）。

## 配布の想定
- 端末配布: MDM でゲートウェイ（Python 同梱の実行形式に固める予定）と環境変数 `OPENAI_BASE_URL` を配る。ブラウザ利用（ChatGPT の Web 画面）は対象外なので、別途ブラウザ拡張か、既存のセキュア Web ゲートウェイで対応する。
- 部門共用: お客様の VPC や社内サーバーに 1 台置き、`--host 0.0.0.0` で受ける（TLS 終端はリバースプロキシで）。
- GPU がない端末: CPU でも動く。`CUDA_VISIBLE_DEVICES= ... --device cpu`（この機の CPU で 1 要求 70〜90 ms を実測）。GPU が他プロセスで埋まっている機では CUDA 初期化で落ちるので、CPU 指定時は GPU を隠して起動する。

## 試作の制限
- HTTPS の上流に対する証明書検証は標準ライブラリ既定。社内プロキシ経由は未対応。
- Anthropic API など OpenAI 互換でない形式は `/v1/messages` の本文キー（content）が同名のため文章の伏せ字は動くが、ヘッダ転送は `x-api-key` / `anthropic-version` のみ対応。
- 画像・ファイル添付の中身は検査しない。
- 伏せ字の復元は同一要求内のみ（会話をまたぐ復元は行わない）。
