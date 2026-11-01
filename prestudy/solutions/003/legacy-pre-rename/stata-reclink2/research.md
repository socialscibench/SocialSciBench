# Competitor: stata-reclink2

## Purpose

The original command, wrapped as a task 008 candidate so it can be measured in the same session,
under the same evaluator protocol, as `bigram-table-scan`: Stata `reclink2` (Wasi and Flaaen
2015, Stata Journal dm0082), run as in `task.md` and `baselines/008_recordlinkage/bench.py`, via
pystata in Stata 18 MP with one processor. The baseline timed it on an Apple M3 Pro with 12 cores
at 134 s on W3 (W4 and W5 estimated at 46 min and 5.4 h); the 14 October study did not run it.

The baseline wrote the using file to disk before its timer; `reclink2` reads its using file from
disk, so here the using records are copied into Stata and saved to a temporary `.dta` inside the
timed call. The kept pairs are the records with a using id; the perfect-pair count is the number
of returned pairs whose records are equal on all five fields.

## Evaluator runs

- `runs/20261019T024248.349045Z-a8f85193`: `--smoke` **failed** on `dev-ties-missing-perfect`
  (the other three fixtures passed): 7 pairs against the reference's 9; the perfect count (2)
  agreed. The fixture's using file holds two identical records ("BETA CORP, 2 MAIN ST, CITY,
  20000, CA", rows 2 and 3) that are the best fuzzy candidate of two master records ("BETA LLC"
  and "ALPHA LTD"). The reference returns both copies for each master record; `reclink2` returns
  one copy each, (1, 3) and (2, 2), with the same scores.

  The cause is in `reclink2.ado`: it collapses the using file to unique combinations of the five
  variables (`egen ... = group(varlist)`) before matching, and maps the matches back to the original
  using ids with an old-style `merge` on the combination id. When several master records matched
  the same duplicated combination, that merge pairs the duplicate ids with them one to one, so
  pairs are lost; a single master record matching a duplicated combination (here the perfect match
  of "ALPHA INC" to rows 0 and 1) is merged 1:m and keeps all of them.
  Contract version 2 requires equal using records to keep distinct row identities, so the
  development gate fails and the evaluator times no workloads. The standard workloads' using
  records are unique, which is why the baseline found identical pairs on W1-W3. This is recorded as
  a failure; the adapter does not work around it.
