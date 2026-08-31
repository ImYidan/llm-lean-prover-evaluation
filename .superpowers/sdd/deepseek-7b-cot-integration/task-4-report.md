# Task 4 Report: DeepSeek Grouped Generation

## Takeover

I took over after the previous implementer hit model-capacity failure. The only
existing worktree change was the untracked
`technical/tests/test_deepseek_generation.py`. I treated it as untrusted partial
RED work, inspected it, kept the useful happy-path grouped-generation test, and
expanded it to cover resume validation, backend grouping contracts, CLI adapter
selection, vLLM metadata preservation, and pre-import generation limit
validation.

The OpenWolf files referenced by the parent `AGENTS.md` were not present in this
repository checkout, so I proceeded from the task brief and existing code.

## RED Evidence

- `PYTHONPATH=/tmp/llm-lean-pytest-20260831 python3 -m pytest -q technical/tests/test_deepseek_generation.py`
- Result: collection failed with `ModuleNotFoundError: No module named 'technical.src.generation.deepseek'`.
- This was valid RED for the missing grouped DeepSeek generation module and
  recovered the partial test file as usable first failing coverage.

After expanding the test file, the same command still failed at collection for
the same missing module, confirming the new Task 4 tests were still RED before
production implementation.

## Implementation

- Added `technical/src/generation/deepseek.py`.
- Added resumable grouped DeepSeek generation through
  `generate_deepseek_records`.
- Appends raw records to `inference.jsonl` before rewriting normalized
  `full_records.json` and `to_inference_codes.json`.
- Validates existing checkpoint rows as an ordered complete-prefix of the
  expected source/sample ID sequence.
- Rejects checkpoint gaps, wrong offsets, excess rows, and incomplete source
  candidate groups.
- Generates one grouped backend request per source problem and rejects wrong
  request-group counts or candidate counts.
- Stores source IDs, prompt, messages, raw output, `raw_response`,
  assembled code, extraction status, `finish_reason`, and `stop_reason`.
- Normalizes raw rows with the shared `to_inference_record` conversion and
  writes JSON arrays with the shared atomic writer.
- Added `--adapter {goedel,deepseek}` with `goedel` as default.
- Kept Goedel on singleton candidates while passing grouped `n=samples` only
  for the DeepSeek adapter.
- Added pre-import validation for positive sample counts and
  `max_tokens < max_model_len`.
- Updated public pipeline wrappers and the ProofNet incremental CLI default to
  pass `max_model_len - 1` as `max_tokens`, matching the stricter token rule.

No changes were made to `DeepSeek-Prover-V2`.

## GREEN Evidence

- `PYTHONPATH=/tmp/llm-lean-pytest-20260831 python3 -m pytest -q technical/tests/test_deepseek_generation.py`
- Result before final cleanup: `12 passed in 0.10s`.

- `PYTHONPATH=/tmp/llm-lean-pytest-20260831 python3 -m pytest -q technical/tests/test_generation.py technical/tests/test_deepseek_generation.py`
- Result after final cleanup: `20 passed in 0.13s`.

- `PYTHONPATH=/tmp/llm-lean-pytest-20260831 python3 -m compileall -q technical/src technical/tests`
- Result: passed.

- `bash -n technical/pipelines/run_standard.sbatch technical/pipelines/run_proofnet_incremental.sbatch technical/pipelines/run_putnam_chunked.sbatch`
- Result: passed.

## Full Suite Evidence

- `PYTHONPATH=/tmp/llm-lean-pytest-20260831 python3 -m pytest -q technical/tests -m 'not integration'`
- Result: `94 passed, 2 deselected in 2.56s`.

- `PYTHONPATH=/tmp/llm-lean-pytest-20260831 python3 -m pytest -q technical/tests`
- Result after final cleanup: `95 passed, 2 skipped in 2.28s`.

- `LEAN_TEST_WORKSPACE=technical/vendor/mathlib4 LEAN_TEST_REPL_COMMAND='lake env repl' PYTHONPATH=/tmp/llm-lean-pytest-20260831 python3 -m pytest -q technical/tests`
- Initial result: `2 failed, 94 passed`; `lake env repl` attempted dependency
  setup and `git` exited 128.
- Escalated retry after dependency fetch: `2 failed, 94 passed`; the REPL
  package existed but `repl` was not executable.
- `lake build repl` in `technical/vendor/mathlib4`: passed.
- Configured retry after building REPL: `2 failed, 94 passed`; REPL import
  initialization and full-header Mathlib compilation still failed.
- Attempted `lake exe cache get`; interrupted after 156 failed artifact
  downloads and 0 successful downloads, with repeated OpenSSL
  `unregistered scheme` errors.
- Final configured retry after cleanup: `2 failed, 95 passed`; both failures
  remained in `technical/tests/test_integration_lean.py` due the local Mathlib
  workspace not being cache-restored/built sufficiently to import Mathlib.

## Self-Review

- Confirmed `git status --short` showed only intended source, test, and public
  pipeline changes.
- Searched changed public source/test/pipeline files for private paths and
  prohibited attribution markers; no violations were found.
- Confirmed `technical/vendor/mathlib4/.lake` artifacts from Lean setup attempts
  are ignored and not staged.
