# Published evaluation records

Use this directory for small, reviewed publication records that link reports to immutable
run bundles. Local experiments stay under ignored `runs/` directories; task directories
contain benchmark definitions and inputs.

Create a portable bundle with `python -m harness export --run runs/RUN_ID --out NEW_DIRECTORY`.
Upload the complete exported directory as an archive attached to a versioned release or
another durable artifact store. Export retains source snapshots, dependency/build records,
traces and logs, and omits transient output arrays. It can preserve failed runs too.

For each published comparison, commit a small record here containing:

- Task, candidate, contract and evaluator protocol versions.
- Original run IDs, source revision and bundle download links with archive SHA-256 hashes.
- Hardware, installed environment, evaluated workloads, failures and timeouts.
- Absolute timings and speedups with their measurement scope.
- A link to the report interpreting the measurements.

Include the runs used by the report and disclose incomplete or failed evaluations. An
exported bundle or a passed subset is not full-task certification. Uploading and publishing
artifacts is a separate action; the evaluator does not publish automatically.
