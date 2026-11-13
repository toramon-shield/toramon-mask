#!/usr/bin/env bash
# 本番生成（再開可能）。使い方: scripts/run_gen.sh <split> <routes(カンマ区切り)> [workers]
cd "$(dirname "$0")/.."; mkdir -p runs/logs
SPLIT=$1; ROUTES=$2; W=${3:-4}
LOG=runs/logs/gen_${SPLIT}_$(echo $ROUTES | tr , _)_$(date +%Y%m%d_%H%M%S).log
.venv/bin/python -m toramon.llm.generate --jobs data/$SPLIT/jobs.jsonl --out data/$SPLIT/raw --routes $ROUTES --workers $W --timeout 600 > "$LOG" 2>&1
echo "EXIT $?" >> "$LOG"
