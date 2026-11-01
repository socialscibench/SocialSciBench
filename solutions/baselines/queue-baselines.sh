#!/bin/bash
# Baseline measurements of the study (2026-10-23), as executed, strictly after the evaluator queue:
# compare.py (the evaluator's worker; outputs validated against saved one-call reference outputs).
# Every case: a first call and one measured call (--warmups 0 --repeats 1). Run from the repository
# root after creating the reference outputs with make_refs.py (one directory per task):
#   REFS=/path/to/refs OUT=/path/to/results solutions/baselines/queue-baselines.sh
# REFS/001, REFS/002 and REFS/003 hold the reference outputs; OUT receives one JSON-lines file per
# (task, adapter, threads) and the log. Do not run anything else on the host meanwhile, and keep the
# host awake (macOS: prefix the command with `caffeinate -i -s`): system sleep pauses the workers, and
# the timeouts use a clock that stops during sleep.
set -u
cd "$(dirname "$0")/../.."
OUT=${OUT:-baseline-results}
REFS=${REFS:?set REFS to the directory with the saved reference outputs (REFS/001, REFS/002, REFS/003)}
mkdir -p "$OUT"
LOG=$OUT/baseline-queue.log
BPY=solutions/baselines/.venv/py/bin/python
cmp() {  # task solution threads timeout cases...
  local task=$1 sol=$2 th=$3 to=$4; shift 4
  echo "=== $(date -u +%FT%TZ) $task $sol threads=$th timeout=$to cases=$*" >> "$LOG"
  $BPY solutions/baselines/compare.py --task $task --solution solutions/$task/$sol/solution.json --cases "$@" \
    --threads $th --warmups 0 --repeats 1 --timeout $to --refdir "$REFS/$task" \
    --out "$OUT/$task-$sol-t$th.jsonl" --python $BPY >> "$LOG" 2>&1
  echo "=== exit $? $(date -u +%FT%TZ)" >> "$LOG"
}
R001="101 102 103 104 104k5 104k50 105 106 107 108"
SYN001="s1 s3 s4 s6 s7 s8 s9 s10 s11 s12"
# 001: strongest compatible baseline (pyfixest + within) on everything, both budgets
cmp 001 pyfixest-within 1 3600 $R001 $SYN001
cmp 001 pyfixest-within 6 3600 $R001 $SYN001
# 002: R did on everything at one thread
cmp 002 did-r 1 5400 101 102 103 105 106 107 108 104 s1 s2 s3 s4 s5 s6
# 003: reclink2
cmp 003 stata-reclink2 1 3600 102 106 107 103 101 105 s1 s2 s3
# 001: fixest and reghdfe on the real workloads
cmp 001 fixest-r 1 3600 101 102 103 106 107 104 105 108
cmp 001 stata-reghdfe 1 3600 106 101 107 102 103 104 105 108
# 002: Stata csdid and csdid2, fastdid, moderndid on representative workloads
cmp 002 stata-csdid-fra 1 3600 106 101 108 102 s1 s2
cmp 002 stata-csdid2-fra 1 3600 106 101 108 102 103 107 s1 s2 s3
cmp 002 fastdid-r 1 3600 106 101 108 102 103 107 s1 s2 s3
cmp 002 moderndid-py 1 3600 106 101 108 102 103 s1 s2 s3
# six-thread budgets
cmp 001 fixest-r 6 3600 101 102 103 106 107 104 105 108
cmp 001 stata-reghdfe 6 3600 106 101 107 102 103
cmp 002 did-r 6 3600 101 102 103 105 107
cmp 002 moderndid-py 6 3600 106 101 108 102 103
cmp 002 stata-csdid2-fra 6 3600 106 101 108 102 103 107
cmp 002 stata-csdid-fra 6 3600 106 101 108 s1 s2
cmp 003 stata-reclink2 6 3600 102 106 107 s1 s2
# synthetic coverage for fixest and pyfixest's alternating projections
cmp 001 fixest-r 1 3600 s1 s3 s4 s6 s7 s8 s10 s12 s9
cmp 001 pyfixest-map 1 3600 106 107 s1 s6 s10 s4 101
echo "=== BASELINES DONE $(date -u +%FT%TZ)" >> "$LOG"
