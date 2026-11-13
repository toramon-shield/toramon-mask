# Toramon Mask — 日本語の個人情報マスキング OSS

**Toramon Mask** は、日本語の文章から氏名・住所・電話番号・メールアドレス・口座番号・API キーなどを見つけて**マスキング（伏せ字化）する、オープンソースのツール**です。社内の文章を ChatGPT や Claude などのクラウド AI に送る前に、**自分の PC の中だけで**個人情報を伏せられます。

```text
入力 : 山田太郎様、お届け先は東京都千代田区丸の内1-2-3、連絡先は090-1234-5678 です。
出力 : [PERSON_1]様、お届け先は[ADDRESS_1]、連絡先は[PHONE_1] です。
```

- **外部に通信しません**。文章の判定は手元の PC で動く小さな AI モデル（約 15 億パラメータ）と、正規表現などのルールで行います。
- **業務に必要な情報は残します**。会社の代表番号、共用メール（`info@` など）、納期、金額、製品型番などは伏せない方針です（詳しくは [保護方針](docs/annotation_policy.md)）。
- コマンドラインのツール、ローカル API、OpenAI 互換のプロキシ（ゲートウェイ）、ブラウザ拡張の 4 通りで使えます。
- 自分のデータでモデルを追加学習し、評価するための一式（合成データ生成、独立レビュー、評価器）も含みます。
- Apache-2.0 ライセンスの OSS です。商用・社内利用、改変、再配布ができます（[12 章](#12-ライセンスと謝辞)）。

> [!IMPORTANT]
> このリポジトリには **AI モデルの重みファイルは含まれていません**。公開モデル（sumeshi/privacy-filter-jp-GGUF など）は、下のクイックスタートの手順でダウンロードします。
> また、どのモデルも**見逃しはゼロになりません**。重要な文書は人の目でも確認してください。

---

## 目次

1. [しくみ（ざっくり）](#1-しくみざっくり)
2. [用語集](#2-用語集)
3. [動作環境](#3-動作環境)
4. [クイックスタート（約 15 分、GPU なしでも可）](#4-クイックスタート約-15-分gpu-なしでも可)
5. [使い方](#5-使い方)
6. [精度を測る（評価）](#6-精度を測る評価)
7. [自分でモデルを学習する（上級）](#7-自分でモデルを学習する上級)
8. [ディレクトリ構成](#8-ディレクトリ構成)
9. [環境変数の一覧](#9-環境変数の一覧)
10. [よくあるトラブル](#10-よくあるトラブル)
11. [セキュリティについて](#11-セキュリティについて)
12. [ライセンスと謝辞](#12-ライセンスと謝辞)

---

## 1. しくみ（ざっくり）

```text
          あなたの PC の中だけで処理
  ┌────────────────────────────────────────────┐
  │                                            │
文章 ──▶ ① AI モデル ──▶ ② ルール R ──▶ ③ 置換 ──▶ 伏せ字の文章 ──▶ クラウド AI
  │   （個人情報らしい   （電話・URL・    （[PERSON_1]  │
  │     範囲を推定）      API キー等を     などに）     │
  │                      補完・補正）                   │
  └────────────────────────────────────────────┘
```

1. **AI モデル**が、文章のどこからどこまでが個人情報かを推定します（「トークン分類」という方式。文章を生成する AI ではありません）。
2. **共通ルール R**（[`configs/rules_R.yaml`](configs/rules_R.yaml)）が、形の決まった情報（電話番号、URL、API キーなど）の見逃しを補い、範囲のずれを直し、会社の代表番号のように残すべきものを除外します。
3. 見つかった範囲を `[PERSON_1]` のような**伏せ字**に置き換えます。ゲートウェイを使う場合は、AI からの返事に含まれる伏せ字を元の値に戻すこともできます。

検出する 8 種類:

| カテゴリ | 例 | 伏せ字 |
|---|---|---|
| `private_person` 個人名 | 山田太郎、ヤマダ タロウ | `[PERSON_n]` |
| `private_address` 個人の住所 | 東京都千代田区丸の内1-2-3 | `[ADDRESS_n]` |
| `private_phone` 個人の電話番号 | 090-1234-5678 | `[PHONE_n]` |
| `private_email` 個人のメール | taro.yamada@example.co.jp | `[EMAIL_n]` |
| `private_date` 個人に紐づく日付 | 生年月日、入院日 | `[DATE_n]` |
| `account_number` 番号類 | 口座番号、会員番号、マイナンバー | `[ACCOUNT_n]` |
| `private_url` 個人に紐づく URL | パスワード再設定リンク | `[URL_n]` |
| `secret` 認証情報 | パスワード、API キー、トークン | `[SECRET_n]` |

## 2. 用語集

| 用語 | 意味 |
|---|---|
| **PII** | Personally Identifiable Information。個人を特定できる情報のこと |
| **マスキング／伏せ字** | 個人情報を `[PERSON_1]` などの記号に置き換えること |
| **S** | 比較の基準にする公開モデル [sumeshi/privacy-filter-jp-GGUF](https://huggingface.co/sumeshi/privacy-filter-jp-GGUF)（日本語向け）。クイックスタートではこれを使います |
| **O** | OpenAI が公開したモデル [openai/privacy-filter](https://huggingface.co/openai/privacy-filter)（英語中心） |
| **T-O / T-S** | このリポジトリの手順で O / S を追加学習したモデル（Toramon Mask の学習版）。重みは同梱していません |
| **GGUF** | AI モデルを 1 ファイルにまとめた形式。`privacy-filter-jp.cpp` という軽量な C++ ランタイムで動きます |
| **HF 形式** | Hugging Face の標準形式。学習や PyTorch での推論に使います |
| **libpf** | `privacy-filter-jp.cpp` をライブラリにしたもの。Python から GGUF モデルを呼び出すのに使います |
| **span（スパン）** | 文章の中の「何文字目から何文字目まで」という範囲 |
| **MASK / KEEP / REVIEW** | 正解データの区分。伏せる／残す／判断が割れる（学習・採点から除外） |

## 3. 動作環境

| | 最低限（試すだけ） | 学習もする場合 |
|---|---|---|
| OS | Linux（Ubuntu 22.04 / 24.04 で確認）。macOS / WSL2 でも CPU 版は動く見込み | Linux |
| CPU / メモリ | 64bit CPU、メモリ 8 GB 以上 | メモリ 64 GB 以上 |
| GPU | **不要**（CPU 版をビルド） | NVIDIA GPU（CUDA 対応）。開発は DGX Spark（GB10、128 GB 統合メモリ）で実施 |
| ディスク | 約 6 GB（モデル 4.5 GB を含む） | 50 GB 以上 |
| ソフト | `git`、`cmake`（3.20 以上）、C/C++ コンパイラ（`g++`）、Python 3.11 以上、[uv](https://docs.astral.sh/uv/)（推奨） | 上記 + Docker と NVIDIA Container Toolkit |

Ubuntu なら、足りないものは次のコマンドで入ります。

```bash
sudo apt update && sudo apt install -y git cmake build-essential python3 python3-venv curl
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv（Python の高速なパッケージ管理ツール）
```

## 4. クイックスタート（約 15 分、GPU なしでも可）

以下は**すべてリポジトリのフォルダの中で**実行します。

### ステップ 1: リポジトリを取得する

```bash
git clone https://github.com/toramon-shield/toramon-mask.git
cd toramon-mask
```

### ステップ 2: Python の環境を作る

```bash
uv venv --python 3.12 .venv                                  # .venv/ に専用の Python 環境を作る
uv pip install -p .venv/bin/python -r env/requirements-host.txt   # 依存ライブラリ（バージョン固定）
uv pip install -p .venv/bin/python -e .                      # このリポジトリ自体（toramon パッケージ）
```

> uv を使わない場合: `python3 -m venv .venv && .venv/bin/pip install -r env/requirements-host.txt && .venv/bin/pip install -e .`

### ステップ 3: 推論ランタイム（privacy-filter-jp.cpp）を用意する

```bash
scripts/setup_third_party.sh          # 外部リポジトリを固定バージョンで third_party/ に取得し、パッチを当てる
PF_CUDA=OFF env/build_pfshared.sh     # CPU 版をビルド（数分）。成功すると「OK: .../libpf.so」と表示
```

NVIDIA GPU がある場合は、`PF_CUDA=OFF` の代わりに GPU の世代を指定して GPU 版をビルドできます（例: RTX 40 シリーズ）。

```bash
CUDA_ARCH=89 env/build_pfshared.sh    # RTX 30 系 = 86、RTX 40 系 = 89、H100 = 90、DGX Spark = 121
```

### ステップ 4: モデルをダウンロードする（約 4.5 GB）

```bash
scripts/download_models.sh            # models/ に保存し、SHA256（改ざん・破損チェック）まで自動で確認
```

最後に `privacy-filter-jp-f16.gguf: OK` などと表示されれば成功です。

### ステップ 5: 動かしてみる

```bash
echo "山田太郎様、ご注文ありがとうございます。お届け先は東京都千代田区丸の内1-2-3、連絡先は090-1234-5678、メールは taro.yamada@example.co.jp です。担当: 株式会社サンプル商事 営業部" \
  | .venv/bin/python scripts/toramon_cli.py redact --backend S --device cpu \
      --gguf models/sumeshi-privacy-filter-jp-GGUF/privacy-filter-jp-f16.gguf
```

次のように表示されれば完成です（会社名は「残す情報」なので伏せられません）。

```text
[PERSON_1]様、ご注文ありがとうございます。お届け先は[ADDRESS_1]、連絡先は[PHONE_1]、メールは [EMAIL_1] です。担当: 株式会社サンプル商事 営業部
```

> GPU 版をビルドした場合は `--device cpu` を `--device cuda` に変えてください。

### ステップ 6（任意）: テストを実行する

```bash
TORAMON_TEST_DEVICE=cpu .venv/bin/python -m pytest -q     # CPU 版をビルドした場合
.venv/bin/python -m pytest -q                             # GPU 版をビルドした場合
```

`105 passed` のように表示されれば成功です。libpf やモデルがまだない場合、それらを使う統合テストは自動でスキップされます。

## 5. 使い方

以降の例では、長いパスを変数にしておくと楽です。

```bash
GGUF=models/sumeshi-privacy-filter-jp-GGUF/privacy-filter-jp-f16.gguf
DEV=cpu        # GPU 版をビルドした場合は cuda
```

### 5-1. コマンドライン（CLI）

```bash
# 伏せ字にした文章を出力
cat 文書.txt | .venv/bin/python scripts/toramon_cli.py redact --backend S --device $DEV --gguf $GGUF

# どこを何として検出したかを JSON で出力（開始・終了位置、種類、確信度）
cat 文書.txt | .venv/bin/python scripts/toramon_cli.py detect --backend S --device $DEV --gguf $GGUF
```

主なオプション:

| オプション | 意味 |
|---|---|
| `--style asterisk` | `[PERSON_1]` ではなく `****` で伏せる |
| `--no-rules` | ルール R を使わず、モデル単体で判定する（比較用） |
| `--threshold 0.5` | モデルの確信度のしきい値。上げると慎重に（伏せる量が減る）、下げると積極的に伏せる |
| `--backend T-O` / `T-S` | 自分で学習・変換した GGUF を使う（[7 章](#7-自分でモデルを学習する上級)） |

### 5-2. ローカル API（他のプログラムから呼ぶ）

```bash
.venv/bin/python scripts/toramon_cli.py serve --backend S --device $DEV --gguf $GGUF --port 8765
```

別のターミナルから:

```bash
curl -s -X POST http://127.0.0.1:8765/redact -H 'Content-Type: application/json' \
  -d '{"text":"佐藤花子さんの電話は080-1111-2222です。"}'
# → {"status":"ok", "redacted_text":"[PERSON_1]さんの電話は[PHONE_1]です。", "spans":[...], ...}
```

| エンドポイント | 内容 |
|---|---|
| `GET /health` | 動作確認 |
| `POST /detect` | 検出結果（span の一覧）を返す |
| `POST /redact` | 伏せ字にした文章と検出結果を返す |

既定では自分の PC（`127.0.0.1`）からしか接続できません。

### 5-3. ゲートウェイ（OpenAI 互換プロキシ）

ChatGPT の API などを使うツールとクラウドの間に挟むと、**送る文章は自動で伏せ字に、返ってきた答えは元の値に戻して**くれます。

```bash
.venv/bin/python scripts/toramon_gateway.py --backend S --device $DEV --gguf $GGUF \
  --port 8787 --upstream https://api.openai.com

# 使う側: 接続先 URL をゲートウェイに向けるだけ（API キーは今までどおり。ゲートウェイは保存しない）
export OPENAI_BASE_URL=http://127.0.0.1:8787/v1
```

- `http://127.0.0.1:8787/admin` で、伏せた件数などの集計を見られます（原文は記録しません）。
- `--block-labels secret`（既定）: API キーやパスワードを含む要求は、送信そのものを止めます。
- 本物の API に送る前に試したいときは、受け取った文章をそのまま返すダミーの上流を使えます: `.venv/bin/python scripts/mock_upstream.py --port 8799` を起動し、`--upstream http://127.0.0.1:8799` を指定します。

詳細（ブラウザ拡張、クリップボード常駐ツール、社内配布の方法）は [docs/gateway_README.md](docs/gateway_README.md) を見てください。

### 5-4. Python から使う

```python
from toramon.adapters.sumeshi_pf import SumeshiPF
from toramon.filter import ToramonFilter

det = SumeshiPF("models/sumeshi-privacy-filter-jp-GGUF/privacy-filter-jp-f16.gguf", device="cpu")
f = ToramonFilter(det, "S")
res = f.redact("山田太郎様の電話は090-1234-5678です。")
print(res.redacted_text)        # [PERSON_1]様の電話は[PHONE_1]です。
for s in res.spans:
    print(s["start"], s["end"], s["label"])
```

## 6. 精度を測る（評価）

正解付きのデータ（JSONL 形式）に対してモデルを動かし、点数を計算できます。診断用に、sumeshi さんの公開ベンチマークを変換したものを [`data/public_bench/`](data/public_bench/) に同梱しています。

```bash
# 1) モデルで予測を作る
.venv/bin/python scripts/run_model.py --model S --device $DEV --gguf $GGUF \
  --input data/public_bench/sumeshi_eval2.jsonl --out runs/demo/S_eval2.pred.jsonl

# 2) 正解と比べて採点する
.venv/bin/python scripts/evaluate.py --gold data/public_bench/sumeshi_eval2.jsonl --pred runs/demo/S_eval2.pred.jsonl

# 3) ルール R なし（モデル単体）の予測も作り、2 つを比べる（差の 95% 信頼区間をブートストラップで計算）
.venv/bin/python scripts/run_filter.py --backend S --device $DEV --gguf $GGUF \
  --input data/public_bench/sumeshi_eval2.jsonl --out runs/demo/S_model_only.pred.jsonl --config model_only
.venv/bin/python scripts/run_filter.py --backend S --device $DEV --gguf $GGUF \
  --input data/public_bench/sumeshi_eval2.jsonl --out runs/demo/S_rules_all.pred.jsonl --config rules_all
.venv/bin/python scripts/evaluate.py --gold data/public_bench/sumeshi_eval2.jsonl \
  --pred runs/demo/S_rules_all.pred.jsonl --compare runs/demo/S_model_only.pred.jsonl
```

表示される主な指標:

| 指標 | 意味 | 良い方向 |
|---|---|---|
| strict F1 | 範囲と種類が**完全に一致**した割合（適合率と再現率の調和平均） | 高い |
| 全範囲保護率 | 正解の個人情報が、**1 文字も漏れずに**伏せられた割合 | 高い |
| PII 残存文書率 | 個人情報が 1 つでも残ってしまった文書の割合 | 低い |
| 非対象文字マスク率 | 伏せなくてよい文字まで伏せてしまった割合（過剰マスキング） | 低い |
| 負例文書誤検知率 | 個人情報を含まない文書で誤って何かを伏せた割合 | 低い |

> 公開ベンチは作者の方針で正解が付けられているため、このリポジトリの保護方針（[docs/annotation_policy.md](docs/annotation_policy.md)）とは一部で判断が異なります。あくまで動作確認・診断用です。

データ形式（1 行 1 文書）は [`configs/data_schema.json`](configs/data_schema.json) で定義しています。最小の例:

```json
{"id": "doc-1", "text": "山田太郎様", "entities": [{"start": 0, "end": 4, "label": "private_person", "action": "MASK"}]}
```

`start` / `end` は**文字単位**の位置です（バイト単位ではありません）。

## 7. 自分でモデルを学習する（上級）

学習には NVIDIA GPU と Docker が必要です。全体の流れは次のとおりです。

```text
① シナリオ設計 → ② LLM で「プレースホルダ入り文書」を生成 → ③ 合成の値を埋め込む
→ ④ 別の LLM 2 系統で独立レビュー → ⑤ 照合 → ⑥ 追加学習 → ⑦ 評価 → ⑧ GGUF に変換して配布
```

**ポイント**: LLM には `{{PERSON_1}}` のような**プレースホルダ入りの文書**だけを書かせ、実際の氏名や電話番号は**プログラムが架空の値を埋め込みます**（[`src/toramon/datagen/pools.py`](src/toramon/datagen/pools.py)）。そのため正解の位置が正確で、実在の個人情報が入り込みません。

### 7-1. 準備

```bash
scripts/download_models.sh --all          # O（OpenAI）も取得
docker build -f env/Dockerfile -t toramon-mask:0.1 .   # 学習用コンテナ（PyTorch 入り）。先に setup_third_party.sh が必要
./env/run.sh python -c "import torch; print(torch.cuda.is_available())"   # True なら OK

# T-S の起点: sumeshi の GGUF を学習可能な HF 形式に戻す
.venv/bin/python scripts/convert_gguf_to_hf.py --gguf models/sumeshi-privacy-filter-jp-GGUF/privacy-filter-jp-f16.gguf \
  --ref models/openai-privacy-filter --out models/sumeshi-privacy-filter-jp-hf-from-f16
```

> Dockerfile のベースイメージは DGX Spark（ARM64）向けの NVIDIA PyTorch コンテナです。x86_64 の PC では、同じ世代の `nvcr.io/nvidia/pytorch` に差し替えてください。

### 7-2. 学習データを作る

文書の生成とレビューには、次のどれかの LLM を使います（`--routes` で選択）。

| 経路 | 使うもの | 必要な設定 |
|---|---|---|
| `gpt` | [OpenAI Codex CLI](https://github.com/openai/codex)（`codex`） | `codex login` 済みであること |
| `gemini` / `claude` | Antigravity CLI（`agy`） | ログイン済みであること |
| `local` | OpenAI 互換の API サーバー（vLLM、llama.cpp server、Ollama など） | `TORAMON_LLM_URL`、`TORAMON_LLM_MODEL`（必要なら `TORAMON_LLM_API_KEY`） |

まず使う LLM を設定します（例: 手元の OpenAI 互換サーバー）。

```bash
export TORAMON_LLM_URL=http://127.0.0.1:8000/v1/chat/completions
export TORAMON_LLM_MODEL=qwen3-32b          # サーバーで動かしているモデル名に置き換え
```

次の 5 段階を、**学習用（train）と選抜用（dev）の両方**について実行します。下の例は `SPLIT` を変えて 2 回回すだけで済むようにしています。

```bash
ROUTES=local          # 生成担当。codex や agy も使えるなら gpt,gemini,claude,local など
REVIEWERS=local       # レビュー担当。生成担当とは別の経路が望ましい（下の注意を参照）

for SPLIT in train dev; do
  N=$([ $SPLIT = train ] && echo 3000 || echo 500)
  SEED=$([ $SPLIT = train ] && echo 1 || echo 2)
  .venv/bin/python scripts/make_jobs.py --split $SPLIT --n $N --out data/$SPLIT/jobs.jsonl --seed $SEED --routes $ROUTES  # ① ジョブ（何を書かせるか）
  .venv/bin/python -m toramon.llm.generate --jobs data/$SPLIT/jobs.jsonl --out data/$SPLIT/raw --routes $ROUTES           # ② 生成
  .venv/bin/python scripts/fill_docs.py --raw data/$SPLIT/raw --split $SPLIT --pool $SPLIT --out data/$SPLIT/filled.jsonl # ③ 値を埋め込む
  .venv/bin/python -m toramon.llm.review --docs data/$SPLIT/filled.jsonl --out data/$SPLIT/reviews --reviewers $REVIEWERS # ④ レビュー
  .venv/bin/python -m toramon.datagen.reconcile --docs data/$SPLIT/filled.jsonl --reviews data/$SPLIT/reviews \
    --out-docs data/$SPLIT/$SPLIT.jsonl --out-report data/$SPLIT/reconcile_report.json                                    # ⑤ 照合
done

.venv/bin/python scripts/audit_splits.py data/train/train.jsonl data/dev/dev.jsonl --out runs/audit/splits.json  # 分割間の重複・漏れを検査
```

結果は `data/train/train.jsonl`（学習用）と `data/dev/dev.jsonl`（選抜用）です。

- **独立レビューについて**: 書いた本人（同じ LLM）がレビューすると見落としが重なりやすいため、レビュー担当は生成担当と**別の経路**にするのが理想です（例: `ROUTES=local REVIEWERS=gpt`）。上の例のように同じ経路を指定しても動きますが、「独立レビューではない」と警告が出ます。`--reviewers` を省略すると「生成担当以外の全経路」が使われ、`local` は `TORAMON_LLM_URL` を設定したときだけ加わります。
- 学習用（train）・選抜用（dev）・最終評価用（test）は、[`configs/scenario_families.yaml`](configs/scenario_families.yaml) で**シナリオ単位**に分けています。似た文書が別の分割に混ざって点数が水増しされるのを防ぐためです。
- 既存の公開データからも作れます: Dolly 日本語版から（`scripts/make_dolly_jobs.py` → `src/toramon/llm/generate_batch.py` → `scripts/make_dolly_dataset.py`）、NVIDIA Nemotron-PII-Japan から（`scripts/convert_nemotron.py`）。**これらのデータを再配布するときは、元のライセンス（CC BY-SA 3.0 / CC BY 4.0）に従ってください。**

### 7-3. 学習する

```bash
# T-O: OpenAI モデルを起点に（MoE の experts を凍結して軽く学習）
./env/run.sh python -m toramon.train.finetune_hf --base /models/openai-privacy-filter \
  --train data/train/train.jsonl --dev data/dev/dev.jsonl --out runs/my_to --mode experts_frozen --epochs 3 --lr 1e-4

# T-S: sumeshi モデルを起点に（LoRA で学習）
./env/run.sh python -m toramon.train.finetune_hf --base /models/sumeshi-privacy-filter-jp-hf-from-f16 \
  --train data/train/train.jsonl --dev data/dev/dev.jsonl --out runs/my_ts --mode lora --epochs 3 --lr 1e-4
```

コンテナの中では、リポジトリが `/work`、`models/` が `/models` に見えます。結果は `runs/<名前>/model/`（HF 形式）と `train_summary.json`、`dev_eval.json` に保存されます。

### 7-4. 比較・最終評価・配布

```bash
# 複数モデル × ルールあり/なしを一括比較し、report.md / report.csv を作る
.venv/bin/python scripts/run_comparison.py --gold data/dev/dev.jsonl --out runs/dev_compare --title "dev 比較" --baseline S \
  --model S=gguf:$GGUF --model T-O=hf:identity:/work/runs/my_to/model --configs model_only,rules_all

# 最終評価（任意。下の手順で作った test 分割に対して、選んだモデルを「一度だけ」評価する）
#  1) test 分割を作る: 7-2 のループを SPLIT=test（例: N=1000, SEED=3）で実行し、結果を eval/locked/filled.jsonl に置く
mkdir -p eval/locked && cp data/test/test.jsonl eval/locked/filled.jsonl
#  2) 正解を凍結する（人手確認の修正ファイルがあれば --decisions で指定。なければ --allow-unverified）
.venv/bin/python scripts/freeze_locked_test.py --allow-unverified
#  3) 評価条件を凍結する（候補モデル・基準モデル・評価コードのハッシュを configs/frozen_conditions.json に記録）
.venv/bin/python scripts/freeze_conditions.py --candidate T-O=runs/my_to/model --baseline S=$GGUF --reason "dev 比較で選定"
#  4) 凍結した条件で一括評価 → runs/final/report.md（CPU 版 libpf の場合は TORAMON_DEVICE=cpu を付ける）
bash scripts/final_eval.sh

# HF 形式 → GGUF に変換（CLI・ゲートウェイで使える形）し、数値が一致するか検証
bash scripts/convert_hf_to_gguf.sh runs/my_to/model my-to f16
.venv/bin/python scripts/verify_gguf_hf_parity.py --help
```

変換した GGUF は `--backend T-O --gguf runs/gguf/my-to/my-to-f16.gguf` で CLI やゲートウェイから使えます。

評価設計の考え方（最終テストの隔離、受入基準の事前凍結など）は [docs/data_design.md](docs/data_design.md) と [configs/acceptance.yaml](configs/acceptance.yaml) にまとめています。

## 8. ディレクトリ構成

```text
toramon-mask/
├── README.md               この説明書
├── LICENSE / NOTICE        ライセンス（Apache-2.0）と第三者の表示
├── pyproject.toml          Python パッケージ設定
├── src/toramon/            本体のライブラリ
│   ├── filter.py             検出 → ルール → 置換をまとめたフィルタ
│   ├── spans.py              文字位置の変換・置換
│   ├── evaluate.py           評価器（各指標、ブートストラップ信頼区間）
│   ├── api.py                ローカル API（FastAPI）
│   ├── paths.py              パスの既定値（環境変数で変更可）
│   ├── adapters/             モデルごとの呼び出し口（GGUF / OpenAI 公式 / HF 形式）
│   ├── rules/                共通ルール R のエンジン
│   ├── datagen/              合成値の生成・埋め込み・レビュー照合
│   ├── llm/                  データ生成・レビュー用 LLM の呼び出し
│   └── train/                追加学習
├── scripts/                実行用スクリプト（CLI、ゲートウェイ、評価、データ作成、変換など）
├── configs/                ラベル対応表、ルール R、データ形式、シナリオ分割、受入基準、モデルの SHA256
├── prompts/                データ生成・レビュー用のプロンプト
├── docs/                   保護方針、データ設計、ゲートウェイの説明
├── gateway/                ブラウザ拡張、クリップボード常駐ツール、配布用メモ
├── env/                    Dockerfile、ビルドスクリプト、外部リポジトリへのパッチ
├── data/public_bench/      動作確認用の公開ベンチマーク（MIT、sumeshi さん作成のものを形式変換）
└── tests/                  テスト
```

次のフォルダはスクリプトが作ります。`.gitignore` 済みなので、GitHub には上がりません。

| フォルダ | 中身 |
|---|---|
| `.venv/` | Python の環境 |
| `third_party/privacy-filter-jp.cpp/`、`third_party/openai-privacy-filter-repo/` | 外部リポジトリ（`setup_third_party.sh`） |
| `models/` | ダウンロードしたモデル（`download_models.sh`） |
| `data/train/` などの生成データ、`runs/`、`checkpoints/`、`eval/` | 生成データ、学習結果、ログ、最終評価用データ |

## 9. 環境変数の一覧

| 変数 | 既定値 | 用途 |
|---|---|---|
| `TORAMON_MODELS_DIR` | `./models` | モデルの置き場所 |
| `TORAMON_PFJP_DIR` | `./third_party/privacy-filter-jp.cpp` | privacy-filter-jp.cpp の場所 |
| `PF_CUDA` | `ON` | `OFF` で CPU 版をビルド（`env/build_pfshared.sh`） |
| `CUDA_ARCH` | `121` | GPU 版ビルド時の GPU 世代 |
| `TORAMON_TEST_DEVICE` | `cuda` | 統合テストで使うデバイス（`cpu` / `cuda`） |
| `TORAMON_DEVICE` | `cuda` | `run_comparison.py`／`final_eval.sh` で GGUF モデルを動かすデバイス |
| `TORAMON_LLM_URL` | `http://127.0.0.1:8000/v1/chat/completions` | データ生成の `local` 経路の接続先 |
| `TORAMON_LLM_MODEL` | `local-model` | `local` 経路のモデル名 |
| `TORAMON_LLM_API_KEY`／`TORAMON_LLM_API_KEY_FILE` | なし | `local` 経路の API キー（またはキーを書いたファイル） |
| `TORAMON_MODEL_GPT`／`TORAMON_MODEL_GEMINI`／`TORAMON_MODEL_CLAUDE` | ソース参照 | CLI 経路で使うモデル ID |
| `TORAMON_IMAGE` | `toramon-mask:0.1` | `env/run.sh` が使う Docker イメージ |

## 10. よくあるトラブル

| 症状 | 原因と対処 |
|---|---|
| `libpf.so` が見つからない | ステップ 3 が終わっていません。`scripts/setup_third_party.sh` → `env/build_pfshared.sh` を実行してください |
| `pf_load failed: no usable 'cuda' device` | CPU 版をビルドしたのに `--device cuda` で動かしています。`--device cpu` にするか、GPU 版をビルドし直してください |
| `cmake` のエラーで止まる | `runs/logs/pfshared_build.log` の末尾を見てください。`cmake` が古い、または `g++` が入っていないことが多いです |
| GPU 版のビルドで `unsupported gpu architecture` | `CUDA_ARCH` があなたの GPU と合っていません（RTX 30 系 = 86、RTX 40 系 = 89 など） |
| `download_models.sh` で `hf: command not found` | ステップ 2 が終わっていません（`.venv/bin/hf` を使います） |
| SHA256 の検証で `FAILED` | ダウンロードが途中で切れています。該当ファイルを消してもう一度実行してください |
| 伏せてほしいものが残る／伏せすぎる | `--threshold` を調整するか、ルール R（`configs/rules_R.yaml`）に規則を追加してください。業務文書での判断基準は [docs/annotation_policy.md](docs/annotation_policy.md) を参照 |
| 長い文書が遅い | CPU 版は GPU 版より遅くなります。長い文書は内部で区切って処理するため、そのまま渡せますが時間はかかります |

## 11. セキュリティについて

- このツールは**補助**です。見逃し（false negative）は起こり得ます。機密性の高い文書は人の目でも確認してください。
- ゲートウェイの監査ログ（既定 `~/.toramon/gateway_audit.jsonl`）には件数と種類だけを記録し、原文は保存しません。
- API キーなどは**環境変数で渡し**、リポジトリにはコミットしないでください（`.env` や `*.key` は `.gitignore` 済みです）。
- このリポジトリのテストやルールには `sk-test-...` や `AKIA...EXAMPLE` のような**ダミーの鍵**が含まれます。すべて架空の値です。
- 脆弱性を見つけた場合は、公開の Issue ではなく GitHub の Security Advisory（非公開の報告）でお知らせください。

## 12. ライセンスと謝辞

- このリポジトリのコードは [Apache License 2.0](LICENSE) で公開しています。
- モデルの重みと外部ソースは同梱していません。ダウンロードされるものは、それぞれのライセンスに従います（[NOTICE](NOTICE) に一覧）。
  - [openai/privacy-filter](https://huggingface.co/openai/privacy-filter)（Apache-2.0）
  - [sumeshi/privacy-filter-jp-GGUF](https://huggingface.co/sumeshi/privacy-filter-jp-GGUF)（Apache-2.0）
  - [sumeshi/privacy-filter-jp.cpp](https://github.com/sumeshi/privacy-filter-jp.cpp)（MIT）。`env/patches/` の小さな修正（DGX Spark の統合 GPU 対応など）も MIT です
- `data/public_bench/` は privacy-filter-jp.cpp のベンチマーク（MIT）を形式変換したものです。
- 日本語 PII の検出モデルとランタイム、ベンチマークを公開してくださった sumeshi さん、ベースモデルを公開した OpenAI と OpenMed の皆さんに感謝します。
