#!/bin/bash
# Follow-up checks of the study (2026-10-23), run after queue-baselines.sh and the adapters' fixture
# checks, sequentially and with nothing else running; development inputs only. They explain the
# reclink2 and csdid failures, check whether the 0.995 trimming rule binds on 002/107 (where the
# baselines without trimming fail), add reclink2 on 003/104 and repeat one measurement that
# overlapped a host sleep. Run from the repository root with system sleep prevented (macOS):
#   REFS=/path/to/refs OUT=/path/to/results caffeinate -i -s solutions/baselines/followup.sh
set -u
cd "$(dirname "$0")/../.."
REFS=${REFS:?set REFS to the directory with the saved reference outputs (REFS/002, REFS/003)}
OUT=${OUT:?set OUT to the results directory}
BPY=solutions/baselines/.venv/py/bin/python
LOG=$OUT/followup.log
step() { echo "=== $(date -u +%FT%TZ) $*" >> "$LOG"; "$@" >> "$LOG" 2>&1; echo "=== exit $? $(date -u +%FT%TZ)" >> "$LOG"; }
# rerun: the queue's measurement of 001/104 with fixest at six threads spanned a host sleep
# (2026-10-23 13:23-16:49 PDT; listed in exclusions.json and left out of the tables)
step $BPY solutions/baselines/compare.py --task 001 --solution solutions/001/fixest-r/solution.json \
  --cases 104 --threads 6 --warmups 0 --repeats 1 --timeout 3600 --refdir "$REFS/001" \
  --out "$OUT/001-fixest-r-t6.jsonl" --python $BPY
step $BPY solutions/baselines/followup_reclink2.py 103 "$REFS/003" "$OUT/followup" --trace
step $BPY solutions/baselines/followup_reclink2.py 102 "$REFS/003" "$OUT/followup"
step $BPY solutions/baselines/followup_reclink2.py 105 "$REFS/003" "$OUT/followup"
step $BPY solutions/baselines/followup_csdid.py 108 "$REFS/002" "$OUT/followup"
step $BPY solutions/baselines/followup_csdid.py 107 "$REFS/002" "$OUT/followup" --numpy-only
step $BPY solutions/baselines/compare.py --task 003 --solution solutions/003/stata-reclink2/solution.json \
  --cases 104 --threads 1 --warmups 0 --repeats 1 --timeout 3600 --refdir "$REFS/003" \
  --out "$OUT/003-stata-reclink2-t1.jsonl" --python $BPY
echo "=== FOLLOWUP DONE $(date -u +%FT%TZ)" >> "$LOG"
