#!/usr/bin/env bash
# Submit with site-specific Slurm resources: sbatch <options> this-file ...
set -euo pipefail

if (( $# != 7 && $# != 9 )); then
  echo "usage: $0 MODEL_CONFIG RUN_CONFIG INPUT_PATH OUTPUT_DIR MODEL_PATH WORKSPACE REPL_COMMAND [CHUNKS CHUNK_INDEX]" >&2
  exit 2
fi

MODEL_CONFIG=$1
RUN_CONFIG=$2
INPUT_PATH=$3
OUTPUT_DIR=$4
MODEL_PATH=$5
WORKSPACE=$6
REPL_COMMAND=$7
PYTHON_BIN=${PYTHON_BIN:-python}

readarray -t SETTINGS < <("${PYTHON_BIN}" -c '
import sys
from pathlib import Path
from technical.src.config import load_config, validate_model_run_config
model = load_config(Path(sys.argv[1]))
run = load_config(Path(sys.argv[2]))
validate_model_run_config(model, run)
if run["generation"]["sample_schedule"] != [32]:
    raise ValueError("autoregressive release runs require sample_schedule: [32]")
for value in (
    model["model"]["adapter"], model["generation"]["seed"],
    model["generation"]["temperature"], model["generation"]["top_p"],
    model["generation"]["top_k"], model["model"]["dtype"],
    model["generation"]["gpu_memory_utilization"],
    run["generation"]["max_model_len"], run["generation"]["max_tokens"],
    model["model"]["tensor_parallel_size"],
    str(model["model"]["trust_remote_code"]).lower(),
    model["model"]["revision"], model["model"]["tokenizer_revision"],
    run["verification"]["timeout"], run["verification"].get("workers", 1),
    run["verification"].get("memory_limit_gb", 0),
    run["verification"].get("restart_every", 0),
    run["verification"]["mode"], run["assembly"]["mode"],
    run["lean"]["profile"],
): print(value)
' "${MODEL_CONFIG}" "${RUN_CONFIG}")

ADAPTER=${SETTINGS[0]}
SEED=${SETTINGS[1]}
TEMPERATURE=${SETTINGS[2]}
TOP_P=${SETTINGS[3]}
TOP_K=${SETTINGS[4]}
DTYPE=${SETTINGS[5]}
GPU_MEMORY_UTILIZATION=${SETTINGS[6]}
MAX_MODEL_LEN=${SETTINGS[7]}
MAX_TOKENS=${SETTINGS[8]}
TENSOR_PARALLEL_SIZE=${SETTINGS[9]}
TRUST_REMOTE_CODE=${SETTINGS[10]}
REVISION=${SETTINGS[11]}
TOKENIZER_REVISION=${SETTINGS[12]}
PROOF_TIMEOUT=${SETTINGS[13]}
WORKERS=${SETTINGS[14]}
MEMORY_LIMIT_GB=${SETTINGS[15]}
RESTART_EVERY=${SETTINGS[16]}
VERIFY_MODE=${SETTINGS[17]}
ASSEMBLY_MODE=${SETTINGS[18]}
LEAN_PROFILE=${SETTINGS[19]}

RUN_INPUT=${INPUT_PATH}
RUN_OUTPUT=${OUTPUT_DIR}
if (( $# == 9 )); then
  CHUNKS=$8
  CHUNK_INDEX=$9
  RUN_OUTPUT=${OUTPUT_DIR}/chunks/chunk_${CHUNK_INDEX}
  RUN_INPUT=${RUN_OUTPUT}/input.jsonl
  mkdir -p "${RUN_OUTPUT}"
  "${PYTHON_BIN}" -m technical.src.benchmarks.putnam_chunking split \
    --input "${INPUT_PATH}" --output "${RUN_INPUT}" \
    --chunks "${CHUNKS}" --index "${CHUNK_INDEX}"
fi

"${PYTHON_BIN}" -m technical.src.lean_profiles \
  --profile "${LEAN_PROFILE}" --workspace "${WORKSPACE}" \
  --repl-command "${REPL_COMMAND}"

if [[ -f "${RUN_OUTPUT}/COMPLETE" ]]; then
  exit 0
fi

MODEL_ARGS=(--revision "${REVISION}" --tokenizer-revision "${TOKENIZER_REVISION}")
if [[ "${TRUST_REMOTE_CODE}" == "true" ]]; then
  MODEL_ARGS+=(--trust-remote-code)
fi

"${PYTHON_BIN}" -m technical.src.generation.generate \
  --input "${RUN_INPUT}" --model-path "${MODEL_PATH}" \
  --output-dir "${RUN_OUTPUT}" --adapter "${ADAPTER}" \
  --assembly-mode "${ASSEMBLY_MODE}" --samples 32 --seed "${SEED}" \
  --temperature "${TEMPERATURE}" --top-p "${TOP_P}" --top-k "${TOP_K}" \
  --dtype "${DTYPE}" --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" --max-tokens "${MAX_TOKENS}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" "${MODEL_ARGS[@]}"

if [[ "${VERIFY_MODE}" == "full_header_file" ]]; then
  "${PYTHON_BIN}" -m technical.src.verification.verify_full_header_file \
    --input "${RUN_OUTPUT}/to_inference_codes.json" \
    --output "${RUN_OUTPUT}/code_compilation_full_header.json" \
    --workspace "${WORKSPACE}" --timeout "${PROOF_TIMEOUT}" \
    --workers "${WORKERS}" --memory-limit-gb "${MEMORY_LIMIT_GB}"
else
  RECOVERY_ARGS=()
  if [[ "${ADAPTER}" == "kimina" ]]; then
    RECOVERY_ARGS=(--recovery-policy kimina)
  fi
  "${PYTHON_BIN}" -m technical.src.verification.verify_full_header_repl \
    --input "${RUN_OUTPUT}/to_inference_codes.json" \
    --output "${RUN_OUTPUT}/code_compilation_full_header.json" \
    --workspace "${WORKSPACE}" --repl-command lake env "${REPL_COMMAND}" \
    --timeout "${PROOF_TIMEOUT}" --workers "${WORKERS}" \
    --memory-limit-gb "${MEMORY_LIMIT_GB}" --restart-every "${RESTART_EVERY}" \
    "${RECOVERY_ARGS[@]}"
fi

"${PYTHON_BIN}" -m technical.src.evaluation.summarize_passk \
  --input "${RUN_OUTPUT}/code_compilation_full_header.json" \
  --full-records "${RUN_OUTPUT}/full_records.json" \
  --output-dir "${RUN_OUTPUT}/summary"
touch "${RUN_OUTPUT}/COMPLETE"
