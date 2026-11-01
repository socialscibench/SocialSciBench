# Candidate environment and build

The initial requirements.lock is copied from the benchmark's tested environment. Verify
that it fits your Python/platform and update it for the actual candidate environment.

From the candidate directory, install dependencies into the environment used by the evaluator:

```bash
python -m pip install -r requirements.lock
```

The reference starter needs no native build. For a compiled implementation, replace this
paragraph with exact build commands, compiler and system-library versions, build hashes,
and accelerator/runtime requirements. Declare all local build inputs in solution.json.

Dependency declarations and build instructions are recorded; the evaluator does not
install packages or execute these instructions automatically.
