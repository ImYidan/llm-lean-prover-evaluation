# DeepSeek-Prover-V2-7B CoT Implementation Plan

**Goal:** Add reproducible DeepSeek-Prover-V2-7B CoT evaluation for miniF2F, ProofNet, PutnamBench, FATE-M, and FATE-H without publishing model weights, datasets, or run artifacts.

**Architecture:** Keep benchmark parsing, Lean assembly, verification, chunking, and Pass@k shared. Add a typed candidate boundary, a DeepSeek prompt/extraction adapter, and an append-only checkpoint layer for grouped sampling and resume.

**Tech stack:** Python 3.10+, pytest, PyYAML, Transformers, vLLM, Lean 4, Mathlib, Lake, Bash/Slurm.

**Design:** [DeepSeek integration design](../design/deepseek-7b-cot-integration.md)

## Global constraints

- Never add weights, tokenizer files, benchmark data, outputs, caches, logs, local paths, credentials, or cluster settings.
- Keep `--model-path` mandatory. Pin the reviewed remote revision as configuration metadata.
- Keep Goedel behavior and its two JSON-array outputs compatible.
- Reject `max_tokens > max_model_len` before loading a model.
- Write `COMPLETE` after generation, verification, summaries, and manifests succeed.
- Commit with `ImYidan <wangyidan752@gmail.com>` and no attribution trailers.

---

## Task 1: Add the candidate boundary and grouped backend output

**Files:**

- Create: `technical/src/generation/candidates.py`
- Modify: `technical/src/generation/generate.py`
- Modify: `technical/tests/test_generation.py`

**Interfaces:**

- Produce `Candidate(text, finish_reason, stop_reason)`.
- Change the internal backend result from `list[str]` to `list[list[Candidate]]`.
- Preserve `generate_records(...)` as the Goedel-compatible singleton consumer.

- [ ] Add a failing test proving that a singleton candidate still creates the historical Goedel records.

```python
class FakeBackend:
    def generate(self, prompts):
        return [[Candidate("```lean4\ntheorem generated : True := by\n  trivial\n```", "stop", None)]]


def test_goedel_rejects_grouped_candidates(tmp_path):
    backend = FakeBackend()
    backend.generate = lambda prompts: [[Candidate("a"), Candidate("b")]]
    with pytest.raises(RuntimeError, match="one candidate"):
        generate_records(ROWS, tokenizer=FakeTokenizer(), backend=backend,
                         adapter=GoedelPromptAdapter(), output_dir=tmp_path,
                         samples=1)
```

- [ ] Run the focused test and confirm it fails because `Candidate` does not exist.

```bash
pytest -q technical/tests/test_generation.py
```

- [ ] Add the immutable value object.

```python
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Candidate:
    text: str
    finish_reason: str | None = None
    stop_reason: str | int | None = None
```

- [ ] Make `_VllmBackend.generate()` retain every vLLM output and its termination metadata.

```python
def generate(self, prompts: list[str]) -> list[list[Candidate]]:
    requests = self.model.generate(prompts, self.sampling_params)
    return [
        [Candidate(item.text, item.finish_reason, item.stop_reason) for item in request.outputs]
        for request in requests
    ]
```

- [ ] Update `generate_records()` to require one candidate per prepared Goedel request and use `candidate.text`.
- [ ] Add `samples_per_prompt: int = 1` to `create_vllm_backend()` and pass it to `SamplingParams(n=...)`.
- [ ] Run `pytest -q technical/tests/test_generation.py` and confirm all tests pass.
- [ ] Commit: `refactor: preserve generation candidate metadata`.

## Task 2: Implement safe append-only checkpoints

**Files:**

- Create: `technical/src/generation/checkpoints.py`
- Create: `technical/tests/test_checkpoints.py`

**Interfaces:**

- Produce `load_checkpoint(path) -> list[dict]`.
- Produce `append_checkpoint(path, record) -> None`.
- Produce `index_checkpoint(records) -> dict[str, dict]` keyed by `problem_id`.
- Permit one incomplete final line; reject malformed interior lines, duplicate IDs, and conflicting records.

- [ ] Add tests for a missing file, valid rows, a torn final line, malformed interior JSON, duplicate IDs, and conflicting IDs.

```python
def test_torn_final_line_is_discarded(tmp_path):
    path = tmp_path / "inference.jsonl"
    path.write_text('{"problem_id":"p_g0"}\n{"problem', encoding="utf-8")
    assert load_checkpoint(path) == [{"problem_id": "p_g0"}]


def test_malformed_interior_line_is_rejected(tmp_path):
    path = tmp_path / "inference.jsonl"
    path.write_text('{"problem_id":"p_g0"}\n{bad}\n{"problem_id":"p_g1"}\n')
    with pytest.raises(CheckpointError, match="line 2"):
        load_checkpoint(path)
```

- [ ] Run `pytest -q technical/tests/test_checkpoints.py` and confirm collection fails.
- [ ] Implement UTF-8 JSONL loading with line-numbered errors and final-line recovery.
- [ ] Implement append using one serialized line, `flush()`, and `os.fsync()`.
- [ ] Reject missing or non-string `problem_id` values and repeated IDs in `index_checkpoint()`.
- [ ] Run `pytest -q technical/tests/test_checkpoints.py` and confirm all tests pass.
- [ ] Commit: `feat: add resumable generation checkpoints`.

## Task 3: Add the DeepSeek prompt and proof extraction adapter

**Files:**

- Create: `technical/src/generation/adapters/deepseek.py`
- Modify: `technical/src/generation/adapters/__init__.py`
- Create: `technical/tests/test_deepseek_adapter.py`

**Interfaces:**

- Consume a benchmark row with `lean4_code` and a tokenizer with `apply_chat_template()`.
- Produce `(prompt, messages)` using the detailed-proof-plan instruction.
- Produce the final Lean fenced block or an explicit extraction failure.
- Reject proof text containing `sorry`, `admit`, `apply?`, or `exact?` as standalone Lean tokens.

- [ ] Add prompt preservation, last-fence selection, missing-fence, and blocked-token tests.

```python
def test_extracts_last_lean_fence():
    output = "plan\n```lean4\nexample : False := by sorry\n```\nfinal\n```lean4\ntheorem p : True := by trivial\n```"
    assert extract_deepseek_proof(output) == "theorem p : True := by trivial"


@pytest.mark.parametrize("token", ["sorry", "admit", "apply?", "exact?"])
def test_rejects_blocked_tokens(token):
    with pytest.raises(ProofAssemblyError, match="blocked"):
        extract_deepseek_proof(f"```lean4\ntheorem p : True := by {token}\n```")
```

- [ ] Run `pytest -q technical/tests/test_deepseek_adapter.py` and confirm collection fails.
- [ ] Implement `DeepSeekPromptAdapter.build_prompt()` without changing the supplied theorem header.
- [ ] Implement final-fence extraction with a token-aware blocked-term expression.
- [ ] Reuse `extract_last_lean_block()` and `replace_theorem_body()`; do not duplicate fence parsing or theorem assembly.
- [ ] Run `pytest -q technical/tests/test_deepseek_adapter.py technical/tests/test_proof_extraction.py`.
- [ ] Commit: `feat: add DeepSeek prompt and extraction adapter`.

## Task 4: Add grouped generation, resume validation, and normalization

**Files:**

- Create: `technical/src/generation/deepseek.py`
- Modify: `technical/src/generation/generate.py`
- Modify: `technical/tests/test_generation.py`
- Create: `technical/tests/test_deepseek_generation.py`

**Interfaces:**

- Consume source rows, a tokenizer, a grouped backend, output directory, sample count, and generation offset.
- Append raw records to `inference.jsonl` before rewriting normalized arrays.
- Produce `full_records.json` and `to_inference_codes.json` through the shared atomic writer.
- Resume from a valid prefix and request the missing suffix.

- [ ] Add a fake grouped backend test for stable `_gN` IDs and metadata.

```python
def test_grouped_generation_writes_raw_and_normalized_records(tmp_path):
    rows = [{"problem_id": "p", "lean4_code": "theorem p : True := by sorry"}]
    generate_deepseek_records(rows, tokenizer=FakeTokenizer(), backend=FakeGroupedBackend(2),
                              output_dir=tmp_path, samples=2, generation_offset=8)
    raw = load_checkpoint(tmp_path / "inference.jsonl")
    assert [row["problem_id"] for row in raw] == ["p_g8", "p_g9"]
    assert raw[0]["finish_reason"] == "stop"
    assert len(json.loads((tmp_path / "full_records.json").read_text())) == 2
```

- [ ] Add resume tests for a complete prefix, gaps, an unexpected offset, excess candidates, and backend count mismatch.
- [ ] Run `pytest -q technical/tests/test_deepseek_generation.py` and confirm collection fails.
- [ ] Implement request planning from the expected ordered ID sequence.
- [ ] Validate existing checkpoint rows against source IDs, offsets, and requested sample counts before model generation.
- [ ] Generate one grouped request per source problem and reject the wrong number of request groups or candidates.
- [ ] Store source identifiers, prompt, raw response, assembled code, extraction status, `finish_reason`, and `stop_reason` in each raw row.
- [ ] Normalize raw rows through shared record conversion and atomic JSON writes after each completed source problem.
- [ ] Add CLI selection with `--adapter {goedel,deepseek}`; keep `goedel` as the default.
- [ ] Add validation for positive sample counts and `max_tokens <= max_model_len` before `create_vllm_backend()`.
- [ ] Run `pytest -q technical/tests/test_generation.py technical/tests/test_deepseek_generation.py`.
- [ ] Commit: `feat: add grouped DeepSeek generation`.

## Task 5: Add pinned model and benchmark run profiles

**Files:**

- Modify: `technical/src/config.py`
- Create: `technical/configs/models/deepseek-prover-v2-7b.yaml`
- Create: `technical/configs/runs/deepseek/minif2f.yaml`
- Create: `technical/configs/runs/deepseek/proofnet.yaml`
- Create: `technical/configs/runs/deepseek/putnam.yaml`
- Create: `technical/configs/runs/deepseek/fate-m.yaml`
- Create: `technical/configs/runs/deepseek/fate-h.yaml`
- Modify: `technical/tests/test_config.py`

**Interfaces:**

- Model profile pins ID `deepseek-ai/DeepSeek-Prover-V2-7B`, revision `a8d9e14432b2e8dd9df2a4d4e70f1ba9bc8d9b7b`, BF16, tensor parallel size 1, seed 30, temperature 1.0, top-p 0.95, and GPU memory utilization 0.90.
- Run profiles select sample schedule, context limits, Lean profile, verification mode, and timeout.

- [ ] Add tests that load all five profiles and assert their benchmark-specific limits.

```python
@pytest.mark.parametrize(
    ("name", "samples", "model_len", "new_tokens", "timeout"),
    [("minif2f", [32], 32768, 8192, 300),
     ("proofnet", [32], 40960, 32768, 300),
     ("putnam", [1, 7, 8, 16], 40960, 32768, 300),
     ("fate-m", [32], 32768, 8192, 4000),
     ("fate-h", [32], 32768, 8192, 4000)],
)
def test_deepseek_run_profiles(name, samples, model_len, new_tokens, timeout):
    config = load_config(ROOT / f"technical/configs/runs/deepseek/{name}.yaml")
    assert config["generation"]["sample_schedule"] == samples
    assert config["generation"]["max_model_len"] == model_len
    assert config["generation"]["max_tokens"] == new_tokens
    assert config["verification"]["timeout"] == timeout
```

- [ ] Run `pytest -q technical/tests/test_config.py` and confirm the new cases fail.
- [ ] Add an explicit run-profile schema instead of accepting unchecked mappings.
- [ ] Write the five profiles with no filesystem paths or scheduler resources.
- [ ] Assert Putnam cumulative offsets derive as `0, 1, 8, 16` from `[1, 7, 8, 16]`.
- [ ] Run `pytest -q technical/tests/test_config.py`.
- [ ] Commit: `config: add DeepSeek benchmark profiles`.

## Task 6: Add the Lean 4.28 FATE environment

**Files:**

- Create: `technical/lean/fate-v428/lean-toolchain`
- Create: `technical/lean/fate-v428/lakefile.toml`
- Create: `technical/lean/fate-v428/lake-manifest.json`
- Modify: `technical/tests/test_public_safety.py`
- Modify: `technical/tests/test_integration_lean.py`

**Interfaces:**

- Pin Lean `v4.28.0`, Mathlib `8f9d9cff6bd728b17a24e163c9402775d9e6a365`, and REPL `527590ce2b9f3b5c4a9a1031e5b8fcfb909b9a4a`.
- Keep dependencies as Lake declarations and manifest pins; do not vendor dependency trees.

- [ ] Add tests that read the three files and assert every version pin.
- [ ] Add a safety test rejecting `.lake/`, `.elan/`, compiled objects, and vendored dependency directories.
- [ ] Run `pytest -q technical/tests/test_public_safety.py technical/tests/test_integration_lean.py` and confirm failure.
- [ ] Add the toolchain and Lake configuration.
- [ ] Generate `lake-manifest.json` with `lake update` from `technical/lean/fate-v428`.
- [ ] Run `lake exe repl` with a trivial theorem and the existing integration-test protocol.
- [ ] Run the two focused pytest files.
- [ ] Commit: `build: add pinned Lean 4.28 FATE profile`.

## Task 7: Add DeepSeek pipelines while retaining shared verification

**Files:**

- Create: `technical/pipelines/run_deepseek.sbatch`
- Create: `technical/pipelines/run_deepseek_putnam_chunked.sbatch`
- Modify: `technical/tests/test_public_safety.py`
- Modify: `technical/tests/test_putnam_chunking.py`

**Interfaces:**

- Standard pipeline consumes model config, run config, input, output directory, caller model path, workspace, and REPL command.
- Putnam pipeline keeps exact cumulative stages `1`, `1+7`, `1+7+8`, and `1+7+8+16`.
- All five profiles use the existing full-header REPL verifier; Putnam also reuses merge and progress modules.

- [ ] Add static tests asserting `--adapter deepseek`, `--model-path`, `--max-tokens`, and the shared verifier commands are present.
- [ ] Add a test that no script contains a user home, scratch path, credential, model cache, or weight filename.
- [ ] Run the focused tests and confirm failure.
- [ ] Implement `run_deepseek.sbatch` using run-profile values and `verify_full_header_repl`.
- [ ] Implement the Putnam wrapper by retaining the validated stage table and switching generation to the DeepSeek adapter.
- [ ] Ensure both scripts remove stale `COMPLETE` markers and create them last.
- [ ] Run `bash -n technical/pipelines/run_deepseek.sbatch technical/pipelines/run_deepseek_putnam_chunked.sbatch`.
- [ ] Run `pytest -q technical/tests/test_public_safety.py technical/tests/test_putnam_chunking.py`.
- [ ] Commit: `feat: add DeepSeek evaluation pipelines`.

## Task 8: Document operation and run the release audit

**Files:**

- Create: `docs/deepseek-7b-cot.md`
- Modify: `README.md`
- Modify: `NOTICE`
- Modify: `.gitignore`
- Modify: `technical/tests/test_public_safety.py`

**Interfaces:**

- Document caller-supplied model path, reviewed revision, five profiles, resume rules, output files, Lean versions, and verification commands.
- Credit upstream model and source projects in `NOTICE` without adding contributor metadata.

- [ ] Add safety tests for weight extensions, output markers, caches, local absolute paths, credentials, generated data, and attribution trailers.
- [ ] Run the safety test and confirm the new assertions fail until ignore rules and docs are complete.
- [ ] Write the user guide and concise README entry.
- [ ] Add upstream notices and ignore rules for `inference.jsonl`, normalized run outputs, summaries, `COMPLETE`, model files, tokenizer files, and Lean build directories.
- [ ] Run all tests.

```bash
pytest -q
```

- [ ] Run syntax and whitespace checks.

```bash
bash -n technical/pipelines/*.sbatch
git diff --check
```

- [ ] Scan tracked files and pending commits for prohibited content and attribution.

```bash
git grep -nEi '(/home/|/scratch/|BEGIN (RSA|OPENSSH) PRIVATE KEY|api[_-]?key|tokenizer\.json|\.safetensors|\.bin$)'
git log --format='%H%n%an <%ae>%n%cn <%ce>%n%B' main..HEAD
```

- [ ] Confirm every commit shows only `ImYidan <wangyidan752@gmail.com>` as author and committer.
- [ ] Commit: `docs: document DeepSeek 7B evaluation`.
- [ ] Request independent diff review, apply verified findings, rerun the full audit, and wait for explicit push authorization.
