#!/usr/bin/env bash
# privacy-filter-jp.cpp の libpf を共有ライブラリ（libpf.so）としてビルドする。Python（ctypes）アダプタが使う。
#
#   env/build_pfshared.sh              # GPU（CUDA）版。CUDA_ARCH で GPU 世代を指定（既定 121 = DGX Spark GB10）
#   PF_CUDA=OFF env/build_pfshared.sh  # CPU 版（CUDA がない PC 向け）
#
# 出力: third_party/privacy-filter-jp.cpp/build/cuda-shared/libpf.so（CPU 版も同じ場所に作る）
# ログ: runs/logs/pfshared_build.log
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC=${TORAMON_PFJP_DIR:-$ROOT/third_party/privacy-filter-jp.cpp}
PF_CUDA=${PF_CUDA:-ON}
CUDA_ARCH=${CUDA_ARCH:-121}   # 例: RTX 30 系 = 86、RTX 40 系 = 89、H100 = 90、DGX Spark = 121
JOBS=${JOBS:-$(nproc)}
B=$SRC/build/cuda-shared
LOG=$ROOT/runs/logs/pfshared_build.log
mkdir -p "$(dirname "$LOG")"
[ -d "$SRC" ] || { echo "$SRC がありません。先に scripts/setup_third_party.sh を実行してください" >&2; exit 2; }

CUDA_FLAGS=(-DPF_CUDA=OFF)
[ "$PF_CUDA" = "ON" ] && CUDA_FLAGS=(-DPF_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")

echo "libpf をビルド中（PF_CUDA=$PF_CUDA）… ログ: $LOG"
{
  echo "== configure $(date -Is)"
  cmake -S "$SRC" -B "$B" -DCMAKE_BUILD_TYPE=Release "${CUDA_FLAGS[@]}" \
        -DBUILD_SHARED_LIBS=ON -DPF_BUILD_TESTS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
  echo "== build $(date -Is)" && cmake --build "$B" -j "$JOBS" --target pf pf-cli
  echo "EXIT $? $(date -Is)"
} > "$LOG" 2>&1

# libpf は CMake 上 STATIC 固定なので、PIC でビルドした libpf.a から共有ライブラリを手動で作る（ctypes 用）。
LIBS=(-lggml -lggml-base -lggml-cpu)
LDIRS=(-L"$B/ggml/src" -Wl,-rpath,"$B/ggml/src")
if [ "$PF_CUDA" = "ON" ]; then
  LIBS+=(-lggml-cuda); LDIRS+=(-L"$B/ggml/src/ggml-cuda" -Wl,-rpath,"$B/ggml/src/ggml-cuda")
fi
if g++ -shared -o "$B/libpf.so" -Wl,--whole-archive "$B/libpf.a" -Wl,--no-whole-archive \
     "${LDIRS[@]}" "${LIBS[@]}" -lpthread >> "$LOG" 2>&1; then
  echo "OK: $B/libpf.so"
else
  echo "失敗しました。$LOG の末尾を確認してください" >&2; tail -20 "$LOG" >&2; exit 1
fi
