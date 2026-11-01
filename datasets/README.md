# Local real-data storage

Development inputs and package code/documentation live in Git-ignored `datasets/001/`,
`datasets/002/` and `datasets/003/`. Held-out inputs, their manifests, all supplied reference
answers (including development answers), and validation logs live **outside the repository**,
by default under `~/SocialSciBench-private/`. Raw source downloads are not included.

| Location | Contents | Files | Bytes |
|---|---|---:|---:|
| `datasets/<task>/` | Development data, original code, build scripts and guides | 99 | 436,574,364 |
| `~/SocialSciBench-private/<task>/data/` | Held-out data, reference answers and validation logs | 70 | 1,136,687,240 |

Together these reproduce all 169 pinned package files. Original package guides describe the
upstream combined layout; use the split layout and commands here for this repository.
Never commit the Parquet files, saved answers or local run outputs.

From the repository root (Python 3.11+ and `curl` required):

```bash
# Development inputs plus package code/documentation; no held-out inputs or answer files
python scripts/download_real_data.py
# Download all files, routing held-out inputs/answers/logs to the external private root
python scripts/download_real_data.py --complete-package
# Verify an existing complete download without network requests
python scripts/download_real_data.py --complete-package --verify-only
# Migrate a previous combined download: verify every destination before removing originals
python scripts/download_real_data.py --relocate-private
```

Use `--task 001` (repeatable), `--out /external/development`,
`--private-root /external/private`, or `--jobs 1` as needed. `--out` changes the development
destination only. Private roots inside the repository or development destination are rejected,
including symlink aliases. Downloads
are resumable at the file level; successful files are reused only when size and SHA-256 match
[the pinned index](../data_sources/packages.json). Shared Drive access and quota must remain
available. The downloader follows only Google's public download form, does not require an
API key, and never disables TLS verification. Changed files fail verification rather than
silently replacing a benchmark release.

Ordinary development evaluation defaults to `datasets/<task>/data`. Frozen final evaluation
requires an explicit external root, for example `--data-root ~/SocialSciBench-private/001/data`.
Freeze records and final run directories must also stay outside the repository. Downloads and
checksum verification do not solve any workloads. Supplied answers and validation logs are
archived for provenance and are not consumed by the shared evaluator.

Keep the private directory available only to the maintainer/final evaluator. Use
`harness stage --suite real-development` for independent agents. Moving files does not prevent
an agent running under the same user account from reading them, and the shared Drive links
remain accessible. Enforce access with an external container or runner that exposes only the
staged workspace, omits private directories and credentials, and restricts network retrieval
of held-out packages and prior results. A read-only mount would still permit reading.

See [data provenance](../data_sources/README.md) and [the evaluation guide](../harness/README.md).
