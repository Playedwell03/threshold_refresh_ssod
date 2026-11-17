#!/bin/bash
# COCO train2017을 20개 부분집합으로 나누고(seed 42) 부분집합 1로 초기 레이블러(YOLOv8s scratch)를 학습한다.
# 결과: runs/base/p1_output/{subsets,weights,runs,eval_*.csv}
# 사용법: bash scripts/run_initial.sh [COCO 경로]
set -euo pipefail
COCO=${1:-coco}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python3}
mkdir -p "$ROOT/runs/base"
cd "$ROOT/src"
"$PY" -u p1_data_slicing_and_initial_learning.py --coco-root "$(cd "$ROOT" && realpath "$COCO")" \
  --out-dir "$ROOT/runs/base" --n 20 --seed 42 --no-show 2>&1 | tee "$ROOT/runs/base/run_initial.log"
