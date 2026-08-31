#!/usr/bin/env bash
# Submit with site-specific Slurm resources, for example: sbatch <options> this-file ...
set -euo pipefail

if (( $# != 7 )); then
  echo "usage: $0 MODEL_CONFIG RUN_CONFIG INPUT_PATH OUTPUT_DIR MODEL_PATH WORKSPACE REPL_COMMAND" >&2
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
RUN_MANIFEST=${OUTPUT_DIR}/run_manifest.json

mkdir -p "${OUTPUT_DIR}"
"${PYTHON_BIN}" -m technical.src.run_manifest ensure \
  --manifest "${RUN_MANIFEST}" \
  --model-config "${MODEL_CONFIG}" \
  --run-config "${RUN_CONFIG}" \
  --input "${INPUT_PATH}" \
  --pipeline deepseek-standard \
  --chunks 1

readarray -t SETTINGS < <("${PYTHON_BIN}" -c '
import sys
from pathlib import Path
from technical.src.config import DEEPSEEK_MODEL_ID, load_config, validate_model_run_config
model = load_config(Path(sys.argv[1]))
run = load_config(Path(sys.argv[2]))
validate_model_run_config(model, run)
if model["model"]["id"] != DEEPSEEK_MODEL_ID:
    raise ValueError("MODEL_CONFIG must select DeepSeek-Prover-V2-7B")
if run["verification"]["mode"] != "full_header_repl":
    raise ValueError("run verification.mode must be full_header_repl")
schedule = run["generation"]["sample_schedule"]
if len(schedule) != 1:
    raise ValueError("standard DeepSeek pipeline requires one sample stage")
assembly_mode = run["assembly"]["mode"]
print(schedule[0])
print(model["generation"]["seed"])
print(model["generation"]["temperature"])
print(model["generation"]["top_p"])
print(model["model"]["dtype"])
print(model["generation"]["gpu_memory_utilization"])
print(run["generation"]["max_model_len"])
print(run["generation"]["max_tokens"])
print(model["model"]["tensor_parallel_size"])
print(str(model["model"]["trust_remote_code"]).lower())
print(model["model"]["revision"] or "__NONE__")
print(run["verification"]["timeout"])
print(assembly_mode)
print(run["lean"]["profile"])
' "${MODEL_CONFIG}" "${RUN_CONFIG}")

SAMPLES=${SETTINGS[0]}
SEED=${SETTINGS[1]}
TEMPERATURE=${SETTINGS[2]}
TOP_P=${SETTINGS[3]}
DTYPE=${SETTINGS[4]}
GPU_MEMORY_UTILIZATION=${SETTINGS[5]}
MAX_MODEL_LEN=${SETTINGS[6]}
MAX_TOKENS=${SETTINGS[7]}
TENSOR_PARALLEL_SIZE=${SETTINGS[8]}
TRUST_REMOTE_CODE=${SETTINGS[9]}
REVISION=${SETTINGS[10]}
PROOF_TIMEOUT=${SETTINGS[11]}
ASSEMBLY_MODE=${SETTINGS[12]}
LEAN_PROFILE=${SETTINGS[13]}
MODEL_ARGS=()
if [[ "${TRUST_REMOTE_CODE}" == "true" ]]; then
  MODEL_ARGS+=(--trust-remote-code)
fi
if [[ "${REVISION}" != "__NONE__" ]]; then
  MODEL_ARGS+=(--revision "${REVISION}")
fi

"${PYTHON_BIN}" -m technical.src.lean_profiles \
  --profile "${LEAN_PROFILE}" \
  --workspace "${WORKSPACE}" \
  --repl-command "${REPL_COMMAND}"

if [[ -f "${OUTPUT_DIR}/COMPLETE" ]]; then
  "${PYTHON_BIN}" -m technical.src.run_manifest check-marker \
    --manifest "${RUN_MANIFEST}" \
    --marker "${OUTPUT_DIR}/COMPLETE" \
    --binding kind=standard
  exit 0
fi

"${PYTHON_BIN}" -m technical.src.generation.generate \
  --input "${INPUT_PATH}" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --adapter deepseek \
  --assembly-mode "${ASSEMBLY_MODE}" \
  --samples "${SAMPLES}" \
  --seed "${SEED}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --dtype "${DTYPE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-tokens "${MAX_TOKENS}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  "${MODEL_ARGS[@]}"

"${PYTHON_BIN}" -m technical.src.verification.verify_full_header_repl \
  --input "${OUTPUT_DIR}/to_inference_codes.json" \
  --output "${OUTPUT_DIR}/code_compilation_full_header.json" \
  --workspace "${WORKSPACE}" \
  --repl-command "${REPL_COMMAND}" \
  --timeout "${PROOF_TIMEOUT}"

"${PYTHON_BIN}" -m technical.src.evaluation.summarize_passk \
  --input "${OUTPUT_DIR}/code_compilation_full_header.json" \
  --full-records "${OUTPUT_DIR}/full_records.json" \
  --output-dir "${OUTPUT_DIR}/summary"

"${PYTHON_BIN}" -m technical.src.run_manifest write-marker \
  --manifest "${RUN_MANIFEST}" \
  --marker "${OUTPUT_DIR}/COMPLETE" \
  --binding kind=standard
