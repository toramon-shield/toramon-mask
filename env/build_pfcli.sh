#!/usr/bin/env bash
# sumeshi privacy-filter-jp.cpp (pf-cli) を CUDA backend 付きでビルドする
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC=${TORAMON_PFJP_DIR:-$ROOT/third_party/privacy-filter-jp.cpp}
# GPU の世代に合わせて変更（121 = DGX Spark GB10。例: RTX 40 系は 89、H100 は 90）
CUDA_ARCH=${CUDA_ARCH:-121}
LOG=$ROOT/runs/logs/pfcli_cuda_build.log
mkdir -p "$(dirname "$LOG")"
{
  echo "== configure $(date -Is)"
  cmake -S "$SRC" -B "$SRC/build/cuda" -DCMAKE_BUILD_TYPE=Release -DPF_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH && \
  echo "== build $(date -Is)" && \
  cmake --build "$SRC/build/cuda" -j 16
  echo "EXIT $? $(date -Is)"
} > "$LOG" 2>&1
