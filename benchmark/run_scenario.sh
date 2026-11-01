#!/usr/bin/env bash
# One target scenario: train the listed methods and the continued-training control,
# score every pool, then the selection analysis. Feature validators and validator
# ensembles (benchmark.extra, --write-ensembles) are no longer part of the default run.
# Usage: bash benchmark/run_scenario.sh etis minent mmd dacs
#        LAMS=0.5 EVERY=400 bash benchmark/run_scenario.sh far fda minent_mmd dacs
set -e
SCENARIO=$1; shift
ALGORITHMS="$*"
PY=.venv/Scripts/python.exe
LOG=runs/benchmark/run_$SCENARIO.log
LAMS=${LAMS:-0.1 0.5 1}
EVERY=${EVERY:-200}
POOLS="source_only uda_adabn uda_continue $(printf 'uda_%s ' $ALGORITHMS)"
{
  echo "start $SCENARIO [$ALGORITHMS] lams=[$LAMS] every=$EVERY $(date)"
  $PY -m benchmark.uda train --algorithm continue --scenario $SCENARIO --lams 0 --every $EVERY
  $PY -m benchmark.uda grid --scenario $SCENARIO --algorithms $ALGORITHMS --lams $LAMS --every $EVERY
  for pool in $POOLS; do
    $PY -m benchmark.score run --pool $pool --scenarios source_val $SCENARIO
  done
  PYTHONIOENCODING=utf-8 $PY -m benchmark.analyze --pools $POOLS --name ${SCENARIO}_methods > /dev/null
  echo "scenario done $(date)"
} >> "$LOG" 2>&1
