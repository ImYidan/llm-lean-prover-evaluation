# Pythagoras-Prover-4B

This release covers only the autoregressive
`Pythagoras-LM/Pythagoras-Prover-4B`. All other model variants are omitted, as
are weights, tokenizer files, caches, datasets, responses, logs, checkpoints,
and scores.

## Fixed inputs

- model and tokenizer revision: `6a3fa6f9b677404073027883492d4a0210e473c5`
- runtime: Python 3.12, vLLM 0.23.0, Transformers 5.12.1, PyTorch 2.11.0
- sampling: seed 42, temperature 0.6, top-p 0.95, top-k -1, 32 candidates
- verification: Lean `v4.9.0-rc1`, Mathlib
  `2f65ba7f1a9144b20c8e7358513548e317d26de1`, separately built REPL
  `6592fd3bec6b3b7f8b9d8432e3f4be08451673b9`
- file compiler: 11 workers, 800-second timeout, 36 GiB address-space limit

Create `technical/environment-autoregressive.yml`, then supply the model,
benchmark JSONL, pinned Mathlib workspace, and pinned absolute REPL executable.
Input rows require a stable `problem_id` (or `name`) and complete
`lean4_code` containing the benchmark declaration with its placeholder proof.

## Run

Use `BENCHMARK` as `minif2f`, `proofnet`, `putnam`, `fate-m`, or `fate-h`:

```bash
sbatch [site-options] technical/pipelines/run_autoregressive.sbatch \
  technical/configs/models/pythagoras-prover-4b.yaml \
  technical/configs/runs/pythagoras/BENCHMARK.yaml \
  /path/to/BENCHMARK.jsonl /path/to/output /path/to/model \
  /path/to/pinned/mathlib /absolute/path/to/pinned/repl
```

The pipeline renders the one-message Goedel-style prompt, requests a grouped
32-candidate vLLM response, atomically checkpoints every raw candidate, applies
last-fence or bare-declaration extraction, requires a verbatim declaration
prefix, grafts the proof onto the canonical statement, forces
`maxHeartbeats 0`, compiles with `lake env lean`, and summarizes empirical
Pass@1/8/16/32 prefixes. It rejects `sorry`, `admit`, `apply?`, and `exact?`.

Generated Lean is untrusted executable input. Run it in a credential-free,
network-disabled environment with read-only inputs and disposable outputs.
