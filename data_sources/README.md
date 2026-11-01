# Real-data provenance

`packages.json` pins all 169 original package files by Drive file ID, modification timestamp,
byte size and SHA-256. All 58 Parquet files were also checked against their original package
manifest checksums. The `001/`, `002/` and `003/` directories preserve the supplied package
guides and build scripts unchanged, except that task 003's copies use its current task number and, for blind review, the guides name their authors by role. Those guides include both measured reference timings and
explicitly labelled estimates; they are provenance documents, not certified benchmark results.

| Current task | Original package | Shared folder |
|---|---|---|
| 001 | `reghdfe_real_data_package` | [Drive](https://drive.google.com/drive/folders/1x7qpYbZlqefE8fH2353eixlNx5dHW80b) |
| 002 | `csdid_real_data_package` | [Drive](https://drive.google.com/drive/folders/1757NAHEygRuPBUx9QOwwGyv_4crOz3-w) |
| 003 | `reclink_real_data_package` | [Drive](https://drive.google.com/drive/folders/1v3S43-udGZVQaKkA9i0kYlQ2gY2Gs0Ii) |

Original guides retain their original task labels and instructions. Run archived build commands
from the downloaded package layout, with its `code/` and `build/` siblings, after obtaining the
raw releases and build dependencies. The tracked copies here document lineage; they are not
the runtime loader. Task 002's builders also require task 001's raw-data staging intermediates.

The active task directories contain adapted import paths, a clean workload catalog and a short
`DATA.md` protocol. The reference implementations and synthetic generators were verified
byte-for-byte identical to the supplied packages; since then, only the opening line of task 003's two
files has changed, to its current task number. New shared-evaluator measurements go to
unique `runs/` directories; no historical measurement was rewritten during this integration.

The original packages group workload 001/109 under development; the evaluator deliberately
exposes it only as an unscored compatibility check. Workload 001/206 includes the A–F
development-area records and must not be described as wholly unseen. See each task's `DATA.md`
for split qualifications and features outside the current computational contract.

Local packages are split: development inputs stay in `datasets/<task>/`, while held-out inputs,
all saved reference answers and validation logs stay outside the repository, by default under
`~/SocialSciBench-private/<task>/data/`. Public definitions, source provenance and checksum
indexes remain here. This directory and private files are excluded from independent-agent
staging. Original guides' combined-layout instructions are historical; use
[the current storage and download instructions](../datasets/README.md).
