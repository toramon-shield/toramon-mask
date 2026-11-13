#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."; mkdir -p runs/logs
LOG=runs/logs/docker_build.log
{ echo "== $(date -Is)"; docker build -f env/Dockerfile -t toramon-mask:0.1 . ; echo "EXIT $? $(date -Is)"; docker image inspect toramon-mask:0.1 --format '{{.Id}}' ; } > "$LOG" 2>&1
