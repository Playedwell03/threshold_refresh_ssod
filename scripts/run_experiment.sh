#!/bin/bash
# 논문 실험 하나를 실행한다.
# src/를 runs/<실험이름>/code/로 복사한 뒤 p0_common_api.py의 임계값 설정 4개만 바꿔서 실행한다.
# (논문 실험도 실험마다 이 4개 값만 바꾼 코드 복사본으로 수행했다 - docs/결과_재현.md 참고)
#
# 사용법: bash scripts/run_experiment.sh <실험이름> [COCO 경로] [추가 옵션...]
#   실험이름: A_global_0.1 ~ A_global_0.7
#             B_fixed_norange, B_fixed_0.3-0.5, B_fixed_0.2-0.85
#             C_refresh_norange, C_refresh_0.3-0.5, C_refresh_0.2-0.85
#   추가 옵션은 실험 코드에 그대로 넘긴다. 예: --dry-run --eval-max-images 10 (학습 없이 준비 단계만 확인)
# 사전 조건: scripts/run_initial.sh로 runs/base/p1_output(부분집합 분할, 초기 레이블러)을 만들어 둘 것
set -euo pipefail
EXP=${1:?실험이름을 지정하세요}
COCO=${2:-coco}
EXTRA=("${@:3}")
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python3}
RUN=$ROOT/runs/$EXP

# 실험별 설정: 스크립트 | 전역 conf | 하한 | 상한 | 갱신 주기(None=고정)
case $EXP in
  A_global_0.[1-7]) SCRIPT=p2_1_fixedthr_ext.py; CONF=${EXP#A_global_} ;;
  B_fixed_norange)    SCRIPT=p2_2_pseudo_label_ext.py; LO=0.00; HI=1.00; K=None ;;
  B_fixed_0.3-0.5)    SCRIPT=p2_2_pseudo_label_ext.py; LO=0.30; HI=0.50; K=None ;;
  B_fixed_0.2-0.85)   SCRIPT=p2_2_pseudo_label_ext.py; LO=0.20; HI=0.85; K=None ;;
  C_refresh_norange)  SCRIPT=p2_2_pseudo_label_ext.py; LO=0.00; HI=1.00; K=3 ;;
  C_refresh_0.3-0.5)  SCRIPT=p2_2_pseudo_label_ext.py; LO=0.30; HI=0.50; K=3 ;;
  C_refresh_0.2-0.85) SCRIPT=p2_2_pseudo_label_ext.py; LO=0.20; HI=0.85; K=3 ;;
  *) echo "알 수 없는 실험이름: $EXP" >&2; exit 1 ;;
esac

[ -d "$ROOT/runs/base/p1_output" ] || { echo "runs/base/p1_output이 없습니다. scripts/run_initial.sh를 먼저 실행하세요." >&2; exit 1; }
[ -e "$RUN" ] && { echo "$RUN 이 이미 있습니다. 기존 결과를 덮어쓰지 않도록 중단합니다." >&2; exit 1; }
mkdir -p "$RUN"
cp -r "$ROOT/src" "$RUN/code"
ln -s "$ROOT/runs/base/p1_output" "$RUN/p1_output"

C=$RUN/code/p0_common_api.py
sed -i 's/"fallback_conf": [0-9.]*,/"fallback_conf": 0.4,/' "$C"
if [ "$SCRIPT" = p2_2_pseudo_label_ext.py ]; then
  sed -i "s/\"min_conf_thr\": [0-9.]*,/\"min_conf_thr\": $LO,/; s/\"max_conf_thr\": [0-9.]*,/\"max_conf_thr\": $HI,/; s/\"recalib_every_n_rounds\": [0-9A-Za-z]*,/\"recalib_every_n_rounds\": $K,/" "$C"
  ARGS=()
else
  ARGS=(--fixed-conf "$CONF" --run-name p2_1_output)
fi
grep -n '"fallback_conf"\|"min_conf_thr"\|"max_conf_thr"\|"recalib_every_n_rounds"' "$C"

cd "$RUN/code"
"$PY" -u "$SCRIPT" --coco-root "$(cd "$ROOT" && realpath "$COCO")" --out-dir "$RUN" --n 20 --no-show "${ARGS[@]}" "${EXTRA[@]}" 2>&1 | tee "$RUN/run.log"
