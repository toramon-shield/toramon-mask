# Windows 用（PowerShell）。Visual Studio Build Tools と CMake で libpf をビルドしてから実行する。
# cmake -S third_party\privacy-filter-jp.cpp -B build\win -DPF_CUDA=OFF ; cmake --build build\win --config Release
Set-Location (Join-Path $PSScriptRoot "..\..")
python -m pip install -q pyinstaller
$libpf = Get-ChildItem -Recurse build\win -Filter "pf*.dll" | Select-Object -First 1
python -m PyInstaller --noconfirm --onedir --name toramon-gateway `
  --add-data "configs\rules_R.yaml;configs" --add-data "configs\label_mapping.yaml;configs" `
  --add-binary "$($libpf.FullName);." --paths src scripts\toramon_gateway.py
Write-Host "dist\toramon-gateway\ に出力。GGUF は別途同梱。"
