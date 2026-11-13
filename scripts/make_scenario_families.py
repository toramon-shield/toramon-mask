#!/usr/bin/env python3
"""シナリオ群（scenario_family）を列挙し、seed 固定で train/dev/test に割り当てる。
生成・調整に先立って実行し、結果（configs/scenario_families.yaml）を凍結する。
"""
import random, yaml, sys, datetime
from pathlib import Path

# domain -> [(scenario_id, 説明)]
FAMILIES = {
  "email": [
    ("quote_followup", "見積依頼・回答のビジネスメール（署名ブロック付き）"),
    ("meeting_arrange", "打合せ日程調整メール"),
    ("intro_newhire", "新任担当者の着任挨拶メール（連絡先共有）"),
    ("delivery_notice", "出荷・配送のお知らせメール（配送先・追跡URL）"),
    ("invoice_send", "請求書送付メール（振込先・担当者）"),
    ("apology_delay", "納期遅延のお詫びメール"),
    ("recruit_interview", "採用面接の案内メール（候補者情報）"),
    ("internal_cc_thread", "社内転送スレッド（引用文に第三者の連絡先）"),
    ("newsletter_optout", "メルマガ配信停止依頼（個人メール）"),
    ("password_reset", "パスワード再設定の案内（リンク・一時コード）"),
    ("event_invite", "セミナー招待・参加登録確認"),
    ("partner_intro", "取引先紹介メール（複数人の連絡先）"),
    ("en_ja_mixed", "英日混在の海外取引先メール"),
    ("complaint_escalation", "クレームのエスカレーション連絡"),
  ],
  "chat": [
    ("sales_slack", "営業チームのチャット（顧客名・電話を共有）"),
    ("support_handoff", "サポート担当間の引き継ぎチャット"),
    ("devops_incident", "障害対応チャット（ログ断片・APIキー貼付）"),
    ("hr_dm", "人事と社員のDM（住所変更・口座変更）"),
    ("line_style_customer", "LINE風の顧客とのやりとり（口語・絵文字）"),
    ("field_report", "現場からの短文報告（訪問先住所・担当者）"),
    ("casual_intro", "雑談チャンネルの自己紹介（ニックネーム・誕生日）"),
  ],
  "minutes": [
    ("sales_weekly", "営業週次会議の議事録"),
    ("board_prep", "取締役会準備ミーティングの議事録"),
    ("customer_visit", "顧客訪問記録（面談者・連絡先）"),
    ("project_kickoff", "プロジェクトキックオフ議事録（体制表）"),
    ("hr_committee", "人事委員会（評価・異動の対象者）"),
    ("vendor_review", "ベンダー評価会議"),
    ("townhall_qa", "全社集会の質疑記録（社員名・質問）"),
    ("audit_meeting", "監査対応打合せ（口座・契約番号）"),
  ],
  "inquiry": [
    ("delivery_change", "配送先変更の問い合わせ"),
    ("billing_dispute", "請求金額相違の問い合わせ"),
    ("account_lock", "アカウントロック解除依頼（ID・登録電話）"),
    ("product_defect", "製品不具合の申告（型番・購入者情報）"),
    ("cancel_subscription", "解約手続きの問い合わせ"),
    ("lost_card", "会員カード紛失の連絡"),
    ("reservation_change", "予約変更（氏名・予約番号・日付）"),
    ("privacy_request", "個人情報の開示・削除請求"),
    ("recall_response", "リコール対応の回答（所有者情報）"),
    ("faq_no_pii", "一般的なFAQ問い合わせ（PIIなし負例）"),
    ("spam_like", "営業スパム風の問い合わせ（公開情報のみ）"),
  ],
  "application": [
    ("membership_form", "入会申込書（フリガナ・生年月日・住所）"),
    ("delivery_slip", "配送伝票・納品書"),
    ("event_registration", "イベント参加申込"),
    ("rental_contract_apply", "賃貸申込書（保証人情報含む）"),
    ("insurance_claim", "保険金請求書（保険証記号番号）"),
    ("scholarship_apply", "奨学金申請（学生・保護者）"),
    ("vehicle_registration", "車両登録・免許証番号"),
    ("bank_account_change", "口座振替依頼書"),
    ("visa_support", "在留資格関連書類（在留カード番号）"),
  ],
  "hr": [
    ("onboarding", "入社手続き案内（マイナンバー・年金番号）"),
    ("payroll_change", "給与振込口座変更"),
    ("evaluation_sheet", "人事評価シート"),
    ("resume_screening", "履歴書スクリーニングメモ"),
    ("leave_application", "休暇・休職申請"),
    ("transfer_notice", "異動通知（複数社員）"),
    ("emergency_contact", "緊急連絡先登録"),
    ("retirement_procedure", "退職手続き"),
    ("training_roster", "研修受講者名簿"),
  ],
  "billing": [
    ("invoice_body", "請求書本文（請求先担当者・振込先）"),
    ("payment_reminder", "入金催促"),
    ("refund_process", "返金処理依頼（口座情報）"),
    ("receipt_issue", "領収書発行依頼"),
    ("subscription_renewal", "サブスク更新通知（カード下4桁は対象外）"),
    ("expense_report", "経費精算（個人立替・口座）"),
  ],
  "contract": [
    ("nda_draft", "秘密保持契約ドラフト（署名者）"),
    ("service_agreement", "業務委託契約（個人事業主の住所）"),
    ("lease_terms", "賃貸借契約条項"),
    ("contract_amendment", "契約変更覚書"),
    ("contract_register", "契約管理台帳（契約番号・担当者）"),
  ],
  "json_csv": [
    ("customer_export_csv", "顧客リストCSV断片"),
    ("api_response_json", "ユーザー情報APIレスポンスJSON"),
    ("order_json", "注文データJSON（配送先）"),
    ("employee_csv", "社員名簿CSV"),
    ("webhook_payload", "Webhookペイロード（トークン含む）"),
  ],
  "log_config": [
    ("app_log_pii", "アプリケーションログ（ユーザー名・メール混入）"),
    ("env_file", ".env / 設定ファイル（APIキー・DB接続文字列）"),
    ("http_headers", "HTTPヘッダ・Authorization"),
    ("audit_log", "監査ログ（操作者・顧客ID）"),
    ("crash_report", "クラッシュレポート（端末情報・メール）"),
    ("server_log_clean", "PIIなしサーバログ（負例）"),
  ],
  "other": [
    ("medical_intake", "問診票（受診日・保険証）※ 病歴は拡張トラック"),
    ("school_notice", "学校からの連絡（保護者・児童名）"),
    ("real_estate_listing", "不動産物件情報（所有者連絡先）"),
    ("local_gov_notice", "自治体からの通知（宛名・住所）"),
    ("press_release", "プレスリリース（公開情報のみ、負例）"),
    ("product_manual", "製品マニュアル抜粋（負例・型番）"),
    ("ocr_noisy_form", "OCR風ノイズを含む申込書"),
    ("long_report", "長文レポート（1万文字超、負荷トラック）"),
    ("sns_post", "SNS投稿（ハンドル名・個人URL）"),
    ("delivery_chat_emoji", "配送ドライバーとの絵文字入りチャット"),
  ],
}

def main(seed: int = 20261109):
    rng = random.Random(seed)
    rows = []
    for dom, items in FAMILIES.items():
        for sid, desc in items:
            rows.append({"id": f"{dom}.{sid}", "domain": dom, "description": desc})
    # 業務分野ごとに層化して 70/10/20 に割当（各 domain で最低 1 件は test に）
    for dom in FAMILIES:
        ids = [r for r in rows if r["domain"] == dom]
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, round(n * 0.2)); n_dev = max(1, round(n * 0.1)) if n >= 5 else 0
        for i, r in enumerate(ids):
            r["split"] = "test" if i < n_test else ("dev" if i < n_test + n_dev else "train")
    rows.sort(key=lambda r: r["id"])
    out = {"version": "0.1", "created_at": str(datetime.date.today()), "seed": seed,
           "note": "生成前に凍結。同一 family の文書・派生は同一 split。test family の本文は評価領域にのみ置く。",
           "counts": {s: sum(1 for r in rows if r["split"] == s) for s in ("train", "dev", "test")},
           "families": rows}
    (Path(__file__).resolve().parents[1] / "configs/scenario_families.yaml").write_text(
        yaml.safe_dump(out, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(out["counts"], "total", len(rows))

if __name__ == "__main__":
    main()
