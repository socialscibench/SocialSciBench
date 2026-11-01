#!/bin/bash
# Sequential evaluator runs of the study's candidates (2026-10-22/23), as executed: task 003, 002 and
# 001 at one thread, then the same at six threads. Cases whose single reference call takes more than
# 150 s on the development host get their own invocation with --warmups 0 --repeats 1 (a first call
# and one measured call, reference and candidate alike); all others use the evaluator's defaults
# (synthetic: 1 warmup + 3 measured calls; real: 1 + 5). Run from the repository root:
#   LOG=/path/to/queue.log solutions/baselines/queue-evaluator.sh      (THREADS="1 6" by default)
# Do not run anything else on the host meanwhile, and keep it awake (macOS: prefix the command with
# `caffeinate -i -s`). Runs go to runs/ (one directory per invocation).
set -u
cd "$(dirname "$0")/../.."
PY=solutions/.venv/bin/python
LOG=${LOG:-queue-evaluator.log}
ev() { echo "=== $(date -u +%FT%TZ) $*" >> "$LOG"; $PY -m harness evaluate "$@" >> "$LOG" 2>&1; echo "=== exit $? $(date -u +%FT%TZ)" >> "$LOG"; }
HEAVY="--warmups 0 --repeats 1"
for TH in ${THREADS:-1 6}; do
  S3=solutions/003/block-index/solution.json
  ev --task 003 --solution $S3 --threads $TH --suite real-development --ids 101 102 103 104 105 106 107 --timeout 3600
  ev --task 003 --solution $S3 --threads $TH --suite real-development --ids 108 $HEAVY --timeout 7200
  ev --task 003 --solution $S3 --threads $TH --ids 1 2 3 4 5 --timeout 3600
  S2=solutions/002/cohort-fused/solution.json
  ev --task 002 --solution $S2 --threads $TH --suite real-development --ids 101 102 103 105 106 107 108 --timeout 3600
  ev --task 002 --solution $S2 --threads $TH --suite real-development --ids 104 $HEAVY --timeout 7200
  ev --task 002 --solution $S2 --threads $TH --ids 1 2 3 4 5 6 --timeout 3600
  S1=solutions/001/schur-cg/solution.json
  ev --task 001 --solution $S1 --threads $TH --suite real-development --ids 101 102 103 106 107 --timeout 3600
  ev --task 001 --solution $S1 --threads $TH --suite real-smoke --timeout 3600
  ev --task 001 --solution $S1 --threads $TH --suite real-development --ids 104 $HEAVY --timeout 7200
  ev --task 001 --solution $S1 --threads $TH --suite real-development --ids 104 --n-regressors 5 $HEAVY --timeout 7200
  ev --task 001 --solution $S1 --threads $TH --suite real-development --ids 104 --n-regressors 50 $HEAVY --timeout 7200
  ev --task 001 --solution $S1 --threads $TH --suite real-development --ids 105 $HEAVY --timeout 7200
  ev --task 001 --solution $S1 --threads $TH --suite real-development --ids 108 $HEAVY --timeout 7200
  ev --task 001 --solution $S1 --threads $TH --ids 1 3 4 6 7 8 9 10 12 --timeout 3600
  ev --task 001 --solution $S1 --threads $TH --ids 11 $HEAVY --timeout 10800
done
echo "=== QUEUE DONE $(date -u +%FT%TZ)" >> "$LOG"
