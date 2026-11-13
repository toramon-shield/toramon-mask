# データ設計（生成・分割・レビュー）v0.1

作成 2026-11-09。データ設計（生成・分割・レビュー）の実装方針。件数は初期計画値であり、パイロットの費用・品質を見て確定する。

## 1. 生成方式: プレースホルダ文書 + プログラム挿入

LLM には**プレースホルダ付き文書**（例 `{{PERSON_1}}`, `{{ADDRESS_1}}`）と各プレースホルダの label/action/entity_id を JSON で返させる。
文字位置は LLM に数えさせず、プログラムが固有値プールから値を挿入して位置を確定する（`src/toramon/datagen/fill.py`）。

プレースホルダ書式: `{{<KIND>_<n>[:<variant>]}}`
- KIND: PERSON, PERSON_KANA, PERSON_ROMAJI, PERSON_FAMILY（姓のみ）, ADDRESS, ADDRESS_PARTIAL（市区町村まで）, POSTAL, EMAIL, PHONE, PHONE_INTL, FAX, DATE_BIRTH, DATE_PERSONAL（入社日等）, ACCOUNT, MYNUMBER, LICENSE, PASSPORT, PENSION, INSURANCE, MEMBER_ID, ORDER_ID, URL_PERSONAL, URL_TRACK, SECRET_PASSWORD, SECRET_APIKEY, SECRET_TOKEN
- KEEP 系（方針上残す。負例・紛らわしさ用）: COMPANY, COMPANY_ADDRESS, COMPANY_PHONE（0120 等）, SHARED_EMAIL（info@ 等）, PUBLIC_URL, DATE_BUSINESS（納期・会議日）, PRODUCT_CODE, VERSION, ERROR_CODE, AMOUNT, INVOICE_NO, CORP_NUMBER（法人番号）, PLACE（施設・駅）, PUBLIC_PERSON（歴史上人物等）
- 同じ `_n` は同一実体。`PERSON_1` と `PERSON_KANA_1` / `PERSON_FAMILY_1` は同一人物（entity_id 共有）。
- variant 例: `{{PHONE_1:fullwidth}}`, `{{PHONE_1:nosep}}`, `{{DATE_BIRTH_1:wareki}}`, `{{PERSON_1:kyujitai}}`。

LLM 側の自由度は「文書の構成・文体・文脈・KEEP 系の具体値（型番や金額など）」。MASK 系の値はすべてプログラムが供給する。
これにより (a) 位置が常に正確、(b) 実在個人情報の混入を構造的に防ぐ、(c) 分割間の固有値漏洩をプール単位で制御できる。

### 自然文直接生成トラック（副）
一部（dev/test の 10% 目安）は LLM が自然文を直接書き、引用文字列で注釈する。プログラムが引用を原文と照合（複数出現は前後文脈で一意化、曖昧なら REVIEW）。プレースホルダ方式に出にくい表現（文中に溶け込む住所省略、口語の名前呼び等）を補う。

## 2. 分割の割り当て（生成前に決める）

| 軸 | 割当単位 | 方法 |
|---|---|---|
| scenario_family | 業務シナリオ群（例 `inquiry.delivery_change`, `hr.onboarding`） | 事前に約 120 family を列挙し、train 70% / dev 10% / test 20% に乱数割当（seed 固定）。同 family は同 split |
| template_family | 文書骨格（LLM へ渡す構成指示のバリアント） | family ごとに複数骨格。骨格も split 専属 |
| value_pool | 氏名・住所・番号などの固有値プール | `pool_train` / `pool_dev` / `pool_test` を完全分離（姓・名・市区町村の**組み合わせ**で分離。要素単位の完全分離は求めない。一般化トラックでは test 専用の珍しい姓名・外国人名・旧字体を使う） |
| generator_model | 3 系統 | 各 split にほぼ均等（各 1/3）。1 社を特定 split 専用にしない |

- 派生（言い換え・値差し替え）は同一 `scenario_family` 内に閉じるので自動的に同 split。
- 監査: 完全一致、NFKC 正規化一致、固有値を `{{KIND}}` に戻した骨格一致、文字 5-gram Jaccard ≥ 0.8、value_pool の越境、生成元偏りを `scripts/audit_splits.py` で検査。

## 3. 件数・分布（初期計画）

| split | 文書数 | 内訳 |
|---|---:|---|
| pilot | 300（各生成元 100） | 開発側のみ。費用・採択率・レビュー不一致率の計測 |
| train | 30,000（必要なら 60,000） | 負例 25%、長さ short 20% / normal 65% / long 15% |
| dev | 3,000 | 同上。閾値・ルール・候補選定 |
| test（locked） | 5,000 | 評価担当のみが本文を持つ。人手確認 subset ≥ 1,000 |

カテゴリ配分（PII あり文書あたり平均 3–6 span）: person 25%, address 15%, phone 15%, email 10%, date 10%, account 15%, url 5%, secret 5%。
業務分野: email/署名 20%, chat 10%, minutes 10%, inquiry 15%, application/delivery 15%, hr 10%, billing 8%, contract 5%, json/csv 4%, log/config 3%。
難例・負例の種類は 設計時に定めたリストを `configs/difficulty_axes.yaml` に列挙し、各 family 生成時に 1–3 軸を指定する。

## 4. 生成・レビューの実行経路（確認済み）

| 系統 | 経路 | モデル ID（固定） |
|---|---|---|
| GPT | `codex exec -m gpt-6.1-terra`（Codex CLI 0.167.0, ChatGPT 認証） | gpt-6.1-terra |
| Gemini | `agy --model gemini-3.8-flash-high`（Antigravity CLI 1.5.3） | gemini-3.8-flash-high |
| Claude | `agy --model claude-sonnet-4-6` | claude-sonnet-4-6 |

いずれも CLI 経由。費用は API 従量ではなく利用枠。記録項目: プロンプト版、モデル ID、所要時間、出力トークン概算（文字数/2）、再試行回数、レート制限の発生。温度・seed は CLI から制御不可（記録上 `null`）。
ツールが使えない担当は「保留」とし、別モデルで代役した場合は generator_model に代役名を記録する。

レビュー: 各文書は生成元以外の 2 系統が独立レビュー（生成元・候補ラベル・他レビューを見せない）。照合後、不一致は人手キューへ。評価対象モデルの予測はレビューに渡さない。

## 5. 最終テストの隔離

- test 用の family・template・value_pool は `eval/locked/` 配下に置き、生成・レビューは評価担当ジョブ（別 OS ユーザー `toramon-eval` を推奨。作れない場合は別ディレクトリ＋権限 700 とし、保証が限定的であることを記録）。
- 開発側に渡すのは件数・分布・品質検査結果のみ。候補と評価条件を凍結（`configs/acceptance.yaml`, `scripts/freeze_conditions.py` が書き出す `configs/frozen_conditions.json`）するまで成績を返さない。
