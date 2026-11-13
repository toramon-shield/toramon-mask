# 端末配布の雛形（MDM）

## Windows（Intune、Win32 アプリ）
- 配布物: `toramon-gateway\`（実行形式一式）＋ GGUF（Q8 1.6 GB）＋ `install.ps1`。
- `install.ps1` の要点: `%ProgramFiles%\Toramon` へ展開、ログオン時起動（タスクスケジューラ、ユーザー権限、`toramon-gateway.exe --device cpu --port 8787 --upstream https://api.openai.com --attribute-user os`）、
  ユーザー環境変数 `OPENAI_BASE_URL=http://127.0.0.1:8787/v1` を設定、Edge/Chrome の拡張を `ExtensionInstallForcelist` で強制配布。
- 検出ルール: `%ProgramFiles%\Toramon\toramon-gateway.exe` の存在。アンインストール: タスク削除と環境変数削除。
- 監査ログの回収: 既定の `%USERPROFILE%\.toramon\gateway_audit.jsonl`（原文なし）を Intune のログ収集または社内共有へ定期コピー。中央集計は各端末の `/audit/summary` を管理サーバーが巡回する構成でも可。

## macOS（Jamf）
- pkg で `/Applications/Toramon/` に配置、LaunchAgent（`~/Library/LaunchAgents/com.example.toramon.gateway.plist`）でログイン時起動。
- `launchctl setenv OPENAI_BASE_URL http://127.0.0.1:8787/v1` を LaunchAgent から実行。拡張は構成プロファイル（ExtensionInstallForcelist）で配布。

## GPU のない端末
- `--device cpu` で動作（Q8、1,000 文字で 70〜90 ms をこの機の CPU で実測）。メモリは GGUF の約 1.6 GB ＋ 数百 MB。

## 注意
- 実行形式は対象 OS でビルドし、コード署名（Windows: Authenticode、macOS: notarization）を付けないと配布時に警告が出る。
- この雛形は未検証。初回は 1 台で手動インストール → 10 台のパイロット → 全社の順で。
