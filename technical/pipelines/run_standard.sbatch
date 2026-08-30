#!/usr/bin/env bash
# Submit with site-specific Slurm resources, for example: sbatch <options> this-file ...
set -euo pipefail

if (( $# < 7 || $# > 8 )); then
  echo "usage: $0 MODEL_CONFIG BENCHMARK_CONFIG INPUT_PATH OUTPUT_DIR MODEL_PATH WORKSPACE REPL_COMMAND [HEADER_POLICY]" >&2
  exit 2
fi

MODEL_CONFIG=$1
BENCHMARK_CONFIG=$2
INPUT_PATH=$3
OUTPUT_DIR=$4
MODEL_PATH=$5
WORKSPACE=$6
REPL_COMMAND=$7
HEADER_POLICY=${8:-}
PYTHON_BIN=${PYTHON_BIN:-python}

mkdir -p "${OUTPUT_DIR}"
rm -f "${OUTPUT_DIR}/COMPLETE"
readarray -t SETTINGS < <("${PYTHON_BIN}" -c '
import sys
from pathlib import Path
from technical.src.config import load_config
model = load_config(Path(sys.argv[1]))
benchmark = load_config(Path(sys.argv[2]))
print(model["generation"]["samples_per_problem"])
print(model["generation"]["temperature"])
print(model["generation"]["top_p"])
print(model["generation"]["max_model_len"])
print(model["model"]["tensor_parallel_size"])
print(str(model["model"]["trust_remote_code"]).lower())
print(model["model"]["revision"] or "__NONE__")
if benchmark["verification"]["mode"] != "standard_repl":
    raise ValueError("benchmark verification.mode must be standard_repl")
print(benchmark["verification"]["timeout"])
print("preserve-options" if benchmark["verification"].get("retain_header") else "strip")
' "${MODEL_CONFIG}" "${BENCHMARK_CONFIG}")

SAMPLES=${SETTINGS[0]}
TEMPERATURE=${SETTINGS[1]}
TOP_P=${SETTINGS[2]}
MAX_MODEL_LEN=${SETTINGS[3]}
TENSOR_PARALLEL_SIZE=${SETTINGS[4]}
TRUST_REMOTE_CODE=${SETTINGS[5]}
REVISION=${SETTINGS[6]}
PROOF_TIMEOUT=${SETTINGS[7]}
HEADER_POLICY=${HEADER_POLICY:-${SETTINGS[8]}}
MODEL_ARGS=()
if [[ "${TRUST_REMOTE_CODE}" == "true" ]]; then
  MODEL_ARGS+=(--trust-remote-code)
fi
if [[ "${REVISION}" != "__NONE__" ]]; then
  MODEL_ARGS+=(--revision "${REVISION}")
fi

"${PYTHON_BIN}" -m technical.src.generation.generate \
  --input "${INPUT_PATH}" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --samples "${SAMPLES}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-tokens "${MAX_MODEL_LEN}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  "${MODEL_ARGS[@]}"

"${PYTHON_BIN}" -m technical.src.verification.verify_standard_repl \
  --input "${OUTPUT_DIR}/to_inference_codes.json" \
  --output "${OUTPUT_DIR}/code_compilation_repl.json" \
  --workspace "${WORKSPACE}" \
  --repl-command "${REPL_COMMAND}" \
  --header-policy "${HEADER_POLICY}" \
  --proof-timeout "${PROOF_TIMEOUT}"

"${PYTHON_BIN}" -m technical.src.evaluation.summarize_passk \
  --input "${OUTPUT_DIR}/code_compilation_repl.json" \
  --full-records "${OUTPUT_DIR}/full_records.json" \
  --output-dir "${OUTPUT_DIR}/summary"

touch "${OUTPUT_DIR}/COMPLETE"
