# Task 007: Fuzzy name matching with bigram cosine similarity

## Computation

Fuzzy matching of company names between two lists: for every pair of names (one from each list), compute a similarity score and keep the pairs scoring at or above a threshold. This is the computation behind

```stata
use master.dta, clear       // idm, namem
matchit idm namem using using.dta, idusing(idu) txtusing(nameu)
```

with `matchit`'s defaults (`similmethod(bigram)`, `score(jaccard)`, `threshold(.5)`). It is the standard way to link databases that share no common identifier: patent assignees to Compustat firms, Crunchbase startups to PitchBook, job-posting employers to a firm database, executives' former employers to firm names. Researchers then review the candidate pairs.

The score `matchit` computes by default (from its source code):

1. Split each name into overlapping **bigrams** (2-character pieces), counted with repetition, with spaces and punctuation included and upper/lower case distinguished. "ACME INC" → `AC`, `CM`, `ME`, `E `, ` I`, `IN`, `NC`.
2. score = Σ_g c₁(g) c₂(g) / √(Σ_g c₁(g)² · Σ_g c₂(g)²), where c₁(g) and c₂(g) are the counts of bigram g in the two names.

That is the **cosine similarity** of the two bigram-count vectors. `matchit` calls it `jaccard`, but it is not the Jaccard index other tools mean by that name (shared ÷ union). R's `stringdist` computes the same score with `method = "cosine", q = 2`. Python's popular `rapidfuzz` does not offer it (its scores are edit-distance based), so the Python reference builds it from sparse bigram-count matrices.

**Where the time goes.** There are n_master × n_using pairs. `matchit` builds an index from bigrams to the using names, so it scores only pairs that share at least one bigram. But company names share common bigrams (from suffixes such as INC, CORP, LLC), so in practice most pairs are scored, one at a time, in interpreted Mata loops. R's `stringdist` scores every pair, using compiled code. The reference multiplies sparse matrices, which scores all pairs sharing a bigram in compiled code.

## Inputs

| Name | Shape | Dtype |
|---|---|---|
| master | [n_master] | string (uppercase ASCII company names) |
| using | [n_using] | string |
| threshold | scalar | float64 (0.5) |

## Outputs

| Name | Shape | Dtype | Stata equivalent |
|---|---|---|---|
| i, j | [n_pairs] | int64 | `idm`, `idu` in the data after `matchit` |
| score | [n_pairs] | float64 | `similscore` |

Pair indices are zero-based. Contract version 2 compares the complete pair set after sorting and each corresponding score; row order is irrelevant. Development fixtures also include short and empty strings, using the reference's bigram convention.

## Interface

Provide a function with the same signature and return value as the one in `reference.py`:

```python
run(master, using, threshold=0.5) -> dict(i, j, score)
```

## Workloads

`workloads.py` (in this folder) generates the data deterministically (each workload has a fixed seed); `generate(w)` builds the inputs for each entry `w` in `WORKLOADS`. Company names combine one to three made-up words with a legal suffix. The using list holds noisy versions of some master names (a different suffix, a typo, a dropped word, punctuation, swapped letters) plus unrelated names, so the true links are known.

| # | design | n_master | n_using | pairs compared | true links |
|---|---|---|---|---|---|
| 1 | sanity check | 500 | 500 | 250,000 | 250 |
| 2 | firm database to firm database (e.g. Compustat to ExecuComp) | 5,000 | 5,000 | 25,000,000 | 3,000 |
| 3 | patent assignees to public firms | 50,000 | 10,000 | 500,000,000 | 3,000 |
| 4 | startups to startups (e.g. Crunchbase to PitchBook) | 50,000 | 50,000 | 2,500,000,000 | 25,000 |
| 5 | job-posting employers to firm database | 200,000 | 20,000 | 4,000,000,000 | 10,000 |

## Correctness

An implementation must match `reference.run` on every workload:

| Output | Tolerance |
|---|---|
| complete set of (i, j) pairs, with no duplicates | exact |
| each corresponding score | relative error ≤ 1e-10 |

Scores are ratios of small integers, so many are exactly 0.5 (e.g. 4 ÷ √(8 × 8)). Compute the score as matchit's formula does (dot product ÷ √(product of sums of squares)) so that exact 0.5 scores are kept; computing similarity as 1 − distance can turn them into 0.4999999999999999 and drop them.

## Scoring

Implementations that pass the correctness checks are scored by wall-clock time of the `run` call, relative to `reference.py` on the same machine. Inputs are already in memory; data generation is not timed.

## Reference implementation

`reference.py` (in this folder) is a readable NumPy/SciPy implementation that scores all pairs sharing a bigram with sparse matrix products. It defines what is computed and is not a performance target.

## Not covered in this version

- Name cleaning before matching (removing legal suffixes, punctuation, stop words), and `matchit`'s `weights()` and `stopwordsauto` options.
- Other similarity measures: token-based scores, edit distances (Levenshtein, Jaro–Winkler), phonetic codes, and learned or AI-based matching.
- Blocking (comparing only names in the same industry, country or first letter), which applied work often uses to limit the number of pairs.

## Shared evaluator (contract version 2)

`task.json` identifies the prompt files, generator, validator and tolerances. Inputs are read-only. Outputs must have the documented shapes and numeric types and contain only finite values. Relative errors use `abs(actual - reference) / max(abs(reference), 1e-12)` element by element, including near zero.

Run `python -m harness evaluate --task 007 --solution solutions/007/candidate/solution.json --smoke` from the repository root for public development fixtures. Omit `--smoke` to validate those fixtures and time official workload 1; use `--ids` or `--all-workloads` for wider coverage. Every first-call, warmup and measured output must pass. The score uses the ratio of reference and candidate median call times on the same host. Generation, imports, first-call setup, validation and serialization are excluded; input-dependent preprocessing and conversion to host-ready outputs are included. See `harness/README.md` for the full protocol. Historical baseline results predate this evaluator and are not automatically certified by it.
