# Task 003: Record linkage using names and addresses

## Workload priority

Prioritize performance on `real-development` workloads while maintaining correctness and
measuring performance on synthetic workloads. Report the suites separately. Held-out inputs
are reserved for final evaluation of frozen implementations. See [DATA.md](DATA.md) for the
workload catalog, development coverage, data locations and split rules. The computation,
accuracy requirements and timing boundary below apply to both real and synthetic inputs.

## Computation

Record linkage of firms on several fields at once (name, street address, city, ZIP code, state) when the two files share no identifier. For every record in the master file, find the record in the using file that agrees best across the fields. This is the computation behind

```stata
use master.dta, clear        // idm name addr city zip state
reclink2 name addr city zip state using using.dta, idmaster(idm) idusing(idu) gen(score) ///
    wmatch(10 8 5 5 5) wnomatch(2 3 4 4 4) required(state)
```

with the other options at their defaults (`minscore(.6)`, `minbigram(.6)`, `npairs(1)`, no `manytoone`). The weights are the ones in the example of `help reclink2`. `reclink2` (Wasi & Flaaen 2015) extends Blasnik's `reclink`; without `manytoone` and `npairs()` both compute the same thing (see Correctness for the one difference). Typical uses are linking survey respondents' employers, startups, loan recipients or patent assignees to a business register when names alone are ambiguous, so the address and location must also agree.

What `reclink2` computes, from its source code:

1. **Perfect matches.** Master records equal to a using record on all five fields get score 1. Both records are removed from the rest of the search.
2. **Candidates (blocking).** For each remaining master record, the candidates are the using records that have the same `state` (`required()`) **and** the same value of at least one of `name`, `addr`, `city`, `zip` (or-blocking, the default when four or more fields are given). Empty master values are skipped. In practice most candidates are the firms in the same city.
3. **Score of each candidate.** For each field, a bigram similarity bi between 0 and 1 (below), turned into a match weight M and a non-match weight NM:
   - M += wmatch × (½ bi² + [bi > .6] × 0.679 × (bi − .6)^(1/3)), where 0.679 = .5 / .4^(1/3)
   - NM += wnomatch × (1 if bi < .4; 1 − bi² if bi < .6; (1 − bi)² otherwise), only if the using value is not empty
   - NM += .3 × wnomatch if exactly one of the two values is empty
   - `required()` fields add wmatch to M.

   The score is M ÷ (M + NM).
4. **Best candidate.** Keep each master record's highest-scoring candidate if its score is at least `minscore` (.6). All tied candidates are kept.

The bigram similarity of master value m and using value s (`reclink`'s `bigram1`):

- (number of the len(m) − 1 two-character pieces of m that occur anywhere in s) × 2 ÷ (len(m) + len(s) − 2);
- strings of fewer than 3 characters score min(len) ÷ (len(m) + len(s) − 1) if one contains the other, else 0;
- then a bonus for a common start: score + k × (1 − score) ÷ 10, where k counts the i in 1–4 with the same first i characters (as in the Jaro–Winkler comparator);
- equal strings score 1, an empty using value 0.

Example: m = "ACME INC", s = "ACME CORP". Four of m's seven pieces (AC, CM, ME, "E ") occur in s, so 4 × 2 ÷ (8 + 9 − 2) = 0.533. The first four characters agree (k = 4), so the score is 0.533 + 4 × 0.467 ÷ 10 = 0.720.

`reclink2` stores bi, M, NM and the score as Stata `float`s (single precision), so the reference rounds every intermediate result to single precision too.

**Where the time goes.** `reclink2` is an ado-file loop over master records. For each record it (1) loads that record from a temporary file (`use ... in i`), (2) loads the **whole** using file with an `if` condition that selects the candidates (`use ... if (name == "..." | addr == "..." | ...) & state == "..."`), and (3) computes each bigram score with one `replace` command per two-character piece of the master value. Step 2 reads all n_using records for every master record, so the time grows with n_master × n_using even though only a few hundred candidates are scored. The R port and the reference instead find all candidates at once with joins (`data.table`, pandas) and score all candidate pairs together.

## Inputs

| Name | Shape | Dtype |
|---|---|---|
| master: name, addr, city, zip, state | [n_master] each | string (uppercase ASCII; "" = missing) |
| using: name, addr, city, zip, state | [n_using] each | string; real registers and development fixtures may contain duplicates, with distinct row identities |
| wmatch, wnomatch | [5] | float64 (10 8 5 5 5; 2 3 4 4 4) |
| required | list | `state` |
| minscore, minbigram | scalar | float64 (.6, .6) |

## Outputs

| Name | Shape | Dtype | Stata equivalent |
|---|---|---|---|
| i | [n_pairs] | int64 | `idm` of master records with a match |
| j | [n_pairs] | int64 | `idu` |
| score | [n_pairs] | float32-rounded values | `score` (1 for perfect matches) |
| n_perfect | scalar | integer | number of perfect pairs, including duplicate-record combinations |

Pair indices are zero-based. Contract version 2 compares the complete pair set after sorting, every corresponding score and the perfect-pair count. Row order is irrelevant. Equal using records retain distinct row identities.

## Interface

Provide a function with the same signature and return value as the one in `reference.py`:

```python
run(master, using, vars, wmatch, wnomatch, required, minscore=0.6, minbigram=0.6)
    -> (i, j, score, n_perfect)

# master, using: dicts of equal-length string arrays keyed by variable name
# vars = ["name", "addr", "city", "zip", "state"], wmatch = [10, 8, 5, 5, 5],
# wnomatch = [2, 3, 4, 4, 4], required = ["state"]  (constants in workloads.py)
```

## Synthetic workloads

`workloads.py` (in this folder) generates the data deterministically (each workload has a fixed seed); `generate(w)` builds the inputs for each entry `w` in `WORKLOADS`. The using file is a register of unique firms: a name of one to three made-up words plus a legal suffix, a street address, and a city, ZIP code and state. About 300 firms share a city on average, with lognormal city sizes; larger states have more cities. The master file holds records of some of these firms as another source would record them, plus firms that are not in the register:

- 25% are exact copies (perfect matches);
- the other 75% can differ in several fields: the name is changed with probability .7 (another suffix, a typo, a dropped word, punctuation, swapped letters); the address is missing (8%) or altered (52%: a spelled-out street type, a house-number typo, no street type, or an added suite); the city has a typo (5%); the ZIP code is missing (10%) or different (5%).

| # | design | n_master | n_using | cities | true links | candidate pairs scored |
|---|---|---|---|---|---|---|
| 1 | sanity check | 500 | 2,000 | 6 | 300 | 149,610 |
| 2 | survey employers to business register | 2,000 | 20,000 | 66 | 1,200 | 691,342 |
| 3 | startups to state business registrations | 10,000 | 100,000 | 333 | 5,000 | 4,592,859 |
| 4 | loan recipients to establishment database | 50,000 | 500,000 | 1,666 | 35,000 | 22,719,376 |
| 5 | establishments to establishment database (national) | 100,000 | 2,000,000 | 6,666 | 70,000 | 42,778,502 |

## Correctness

An implementation must match `reference.run` on every workload:

| Output | Tolerance |
|---|---|
| complete set of (i, j) pairs, with no duplicate pairs; n_perfect | exact |
| each corresponding score | relative error ≤ 1e-6 |

Notes for implementers:

- **Single precision.** `reclink2` stores bigram scores, M, NM and the final score as `float`. The reference and the R port round each intermediate result to single precision and reproduce `reclink2`'s scores exactly. This matters more than rounding usually does. A bigram score of exactly .6 (e.g. 3 × 2 ÷ 10) is stored as 0.60000002, which is above `minbigram`, so it earns the cube-root bonus 0.679 × (0.00000002)^(1/3) ≈ 0.002. In double precision it would earn nothing. On workload 3, a double-precision version keeps the same pairs, but individual scores differ by up to 3e-4 and the score sum by 2e-6, which fails the tolerance above. Ties could also break differently.
- **`reclink` vs `reclink2`.** `reclink` keeps a best candidate if its score is **>** `minscore`; `reclink2` if it is **≥** `minscore`. This never matters with the default .6, because a single-precision score cannot equal the double-precision .6 exactly (the nearest float is 0.60000002). So both commands return identical results here.
- **Or-blocking only with four or more variables.** With three or fewer variables and no `orblock()`, every using record that agrees on `required()` is a candidate, so the number of candidates (and the time) grows much faster.

## Scoring

Implementations that pass the correctness checks are scored by wall-clock time of the `run` call, relative to `reference.py` on the same machine. Inputs are already in memory; data generation is not timed.

## Reference implementation

`reference.py` (in this folder) is a readable NumPy/pandas implementation that finds candidates with joins and scores all candidate pairs together. It defines what is computed and is not a performance target.

## Not covered in this version

- The `manytoone` and `npairs()` options of `reclink2`, `exactstr()`, `exclude()` and `orblock()` set by hand.
- Preprocessing (`stnd_compname`, `stnd_address`) and clerical review (`clrevmatch`).
- Other linkage tools with other scoring rules (`dtalink`, R `reclin2` and `fastLink`, Python `recordlinkage`, Splink).

## Shared evaluator (contract version 2)

`task.json` identifies the prompt files, generator, validator and tolerances. Inputs are read-only. Outputs must have the documented shapes and numeric types and contain only finite values. Relative errors use `abs(actual - reference) / max(abs(reference), 1e-12)` element by element, including near zero.

Run `python -m harness evaluate --task 003 --solution solutions/003/candidate/solution.json --smoke` from the repository root for public development fixtures. With the default `synthetic` suite, omit `--smoke` to validate those fixtures and time synthetic workload 1. Use `--ids` or `--all-workloads` for wider coverage within the selected suite; select `--suite real-development` explicitly for real inputs. Every first-call, warmup and measured output must pass. The score uses the ratio of reference and candidate median call times on the same host. Generation, imports, first-call setup, validation and serialization are excluded; input-dependent preprocessing and conversion to host-ready outputs are included. See `harness/README.md` for the full protocol. Historical baseline results predate this evaluator and are not automatically certified by it.

After the smoke check, start real-development evaluation from the repository or staged-workspace root:

```bash
python -m harness evaluate --task 003 --solution solutions/003/candidate/solution.json --suite real-development --ids 102 --threads 1
```

A staged workspace includes only the selected inputs; consult its `stage.json` and [DATA.md](DATA.md).
