#!/usr/bin/env bash
# Runs once the ClinicDB grid has finished: continued-training control, the extra
# validator pass for every pool, then the full analysis with validator ensembles.
set -e
PY=.venv/Scripts/python.exe
LOG=runs/benchmark/after_grid_clinicdb.log
POOLS="source_only uda_adabn uda_minent uda_advent uda_dann uda_mmd uda_bnm uda_mcc uda_meanteacher uda_fda uda_continue"
until grep -q "^real" runs/benchmark/grid_clinicdb.log; do sleep 60; done
{
  $PY -m benchmark.uda train --algorithm continue --scenario clinicdb --lams 0
  for pool in uda_adabn uda_minent uda_advent uda_dann uda_mmd uda_bnm uda_mcc uda_meanteacher uda_fda uda_continue; do
    $PY -m benchmark.score run --pool $pool --scenarios source_val clinicdb
  done
  for pool in $POOLS; do
    $PY -m benchmark.extra --pool $pool --scenarios source_val clinicdb
  done
  $PY -m benchmark.analyze --pools $POOLS --name clinicdb_full --write-ensembles > /dev/null
  $PY -m benchmark.score run --pool ensemble_clinicdb_full --scenarios source_val clinicdb
  PYTHONIOENCODING=utf-8 $PY -m benchmark.analyze --pools $POOLS --name clinicdb_full > /dev/null
  echo "after-grid done $(date)"
} >> "$LOG" 2>&1
