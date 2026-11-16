#!/usr/bin/env bash
# Launch all (method, seed) runs on one GPU, PARALLEL at a time (default 3).
# Environment overrides:
#   OUT=runs DEVICE=cuda:0 PARALLEL=3 EPOCHS=20 METHODS="naive replay ewc noshift" SEEDS="0 1 2"
#   NUM_WORKERS=4 EXTRA="--smoke --fake-data"   (EXTRA is passed verbatim to run_pilot.py)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
OUT=${OUT:-runs}
DEVICE=${DEVICE:-cuda:0}
PARALLEL=${PARALLEL:-3}
EPOCHS=${EPOCHS:-20}
METHODS=${METHODS:-"naive replay ewc noshift"}
SEEDS=${SEEDS:-"0 1 2"}
NUM_WORKERS=${NUM_WORKERS:-4}
EXTRA=${EXTRA:-}
NCPU=$(nproc)
export OMP_NUM_THREADS=$(( NCPU / PARALLEL > 0 ? NCPU / PARALLEL : 1 ))
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$OUT/logs"

if [[ "$EXTRA" != *"--fake-data"* ]]; then
  "$PY" run_pilot.py --download-only --method naive --seed 0 --out "$OUT" --device cpu
fi

echo "$(date '+%F %T') start: methods=[$METHODS] seeds=[$SEEDS] parallel=$PARALLEL epochs=$EPOCHS threads/run=$OMP_NUM_THREADS"
jobs_list=()
for s in $SEEDS; do for m in $METHODS; do jobs_list+=("$m $s"); done; done

printf '%s\n' "${jobs_list[@]}" | xargs -P "$PARALLEL" -L 1 bash -c '
  m=$0; s=$1
  echo "$(date "+%F %T") launch $m seed $s"
  if '"$PY"' run_pilot.py --method "$m" --seed "$s" --epochs-per-task '"$EPOCHS"' \
       --num-workers '"$NUM_WORKERS"' --out '"$OUT"' --device '"$DEVICE"' '"$EXTRA"' \
       > '"$OUT"'/logs/${m}_seed${s}.log 2>&1; then
    echo "$(date "+%F %T") done   $m seed $s"
  else
    echo "$(date "+%F %T") FAILED $m seed $s (see '"$OUT"'/logs/${m}_seed${s}.log)"; exit 255
  fi'

"$PY" aggregate.py --runs "$OUT" --out "$OUT/aggregate"
echo "$(date '+%F %T') all runs finished and aggregated in $OUT/aggregate"
