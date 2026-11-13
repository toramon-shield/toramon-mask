#!/usr/bin/env bash
# Toramon Gateway を単体実行形式にする（PyInstaller）。対象 OS 上で実行する（Windows は build_exe.ps1）。この機（Linux aarch64）では Windows/macOS 版は作れない。
# 前提: privacy-filter-jp.cpp の libpf を対象 OS 用にビルド済み（third_party/privacy-filter-jp.cpp、cmake、CPU ビルドで可）。
set -euo pipefail; cd "$(dirname "$0")/../.."
PY=${PY:-python3}; $PY -m pip install -q pyinstaller
LIBPF=$(ls third_party/privacy-filter-jp.cpp/build/*/libpf.* 2>/dev/null | head -1 || true)
$PY -m PyInstaller --noconfirm --onedir --name toramon-gateway \
  --add-data "configs/rules_R.yaml:configs" --add-data "configs/label_mapping.yaml:configs" \
  ${LIBPF:+--add-binary "$LIBPF:."} \
  --paths src scripts/toramon_gateway.py
echo "dist/toramon-gateway/ に出力。GGUF（runs/gguf/toramon-to-v9s3/*.gguf）は別途同梱し、起動時に --gguf で指定する。"
