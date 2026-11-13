#!/usr/bin/env bash
# 環境診断: ホストの OS/CPU/GPU/ドライバ/CUDA/メモリ/ディスク/ツールチェーンを JSON で記録する
set -u
out=${1:-env/diagnosis_$(date +%Y%m%d_%H%M%S).json}
py() { python3 - "$@"; }
HOST=$(hostname); KERNEL=$(uname -r); ARCH=$(uname -m)
OS=$(. /etc/os-release; echo "$PRETTY_NAME")
CPU=$(lscpu | awk -F: '/Model name/{gsub(/^ +/,"",$2);print $2}' | head -1); NCPU=$(nproc)
GPU=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)
CUDA_SMI=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9.]+')
NVCC=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9.]+')
MEMTOTAL=$(awk '/MemTotal/{print $2}' /proc/meminfo); MEMAVAIL=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
SWAPTOTAL=$(awk '/SwapTotal/{print $2}' /proc/meminfo); SWAPFREE=$(awk '/SwapFree/{print $2}' /proc/meminfo)
DISK=$(df -k --output=size,avail / | tail -1)
GPUPROCS=$(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null | tr '\n' ';')
DOCKER=$(docker --version 2>/dev/null); GCC=$(gcc --version | head -1); CMAKE=$(cmake --version | head -1)
PY=$(python3 --version); UV=$(uv --version 2>/dev/null)
python3 - "$out" <<PY
import json,sys,datetime
d=dict(timestamp=datetime.datetime.now().astimezone().isoformat(),host="$HOST",kernel="$KERNEL",arch="$ARCH",os="$OS",
 cpu="$CPU",ncpu=int("$NCPU"),gpu="$GPU",cuda_driver_api="$CUDA_SMI",nvcc="$NVCC",
 mem_total_gb=round(int("$MEMTOTAL")/2**20,1),mem_available_gb=round(int("$MEMAVAIL")/2**20,1),
 swap_total_gb=round(int("$SWAPTOTAL")/2**20,1),swap_free_gb=round(int("$SWAPFREE")/2**20,1),
 disk_root_total_gb=round(int("$DISK".split()[0])/2**20,1),disk_root_avail_gb=round(int("$DISK".split()[1])/2**20,1),
 gpu_compute_processes="$GPUPROCS",docker="$DOCKER",gcc="$GCC",cmake="$CMAKE",python="$PY",uv="$UV",
 note="DGX Spark unified memory: nvidia-smi does not report GPU memory separately; mem_available_gb is the practical headroom for both CPU and GPU.")
json.dump(d,open(sys.argv[1],"w"),ensure_ascii=False,indent=2); print(json.dumps(d,ensure_ascii=False,indent=2))
PY
