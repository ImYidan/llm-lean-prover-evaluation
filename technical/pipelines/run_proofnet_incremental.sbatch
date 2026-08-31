#!/usr/bin/env bash
# Submit with site-specific Slurm resources, for example: sbatch <options> this-file ...
set -euo pipefail

if (( $# != 6 )); then
  echo "usage: $0 INPUT_PATH OUTPUT_DIR MODEL_PATH WORKSPACE MODEL_CONFIG BENCHMARK_CONFIG" >&2
  exit 2
fi

INPUT_PATH=$1
OUTPUT_DIR=$2
MODEL_PATH=$3
WORKSPACE=$4
MODEL_CONFIG=$5
BENCHMARK_CONFIG=$6
PYTHON_BIN=${PYTHON_BIN:-python}

mkdir -p "${OUTPUT_DIR}"
rm -f "${OUTPUT_DIR}/COMPLETE"

readarray -t SETTINGS < <("${PYTHON_BIN}" -c '
import sys
from pathlib import Path
from technical.src.config import load_config
model = load_config(Path(sys.argv[1]))
benchmark = load_config(Path(sys.argv[2]))
if benchmark["verification"]["mode"] != "full_header_file":
    raise ValueError("benchmark verification.mode must be full_header_file")
print(model["generation"]["temperature"])
print(model["generation"]["top_p"])
print(model["generation"]["max_model_len"])
print(model["model"]["tensor_parallel_size"])
print(str(model["model"]["trust_remote_code"]).lower())
print(model["model"]["revision"] or "__NONE__")
print(benchmark["verification"]["timeout"])
' "${MODEL_CONFIG}" "${BENCHMARK_CONFIG}")

MODEL_ARGS=()
if [[ "${SETTINGS[4]}" == "true" ]]; then
  MODEL_ARGS+=(--trust-remote-code)
fi
if [[ "${SETTINGS[5]}" != "__NONE__" ]]; then
  MODEL_ARGS+=(--revision "${SETTINGS[5]}")
fi

"${PYTHON_BIN}" -m technical.src.benchmarks.run_proofnet_incremental \
  --input "${INPUT_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-path "${MODEL_PATH}" \
  --workspace "${WORKSPACE}" \
  --temperature "${SETTINGS[0]}" \
  --top-p "${SETTINGS[1]}" \
  --max-model-len "${SETTINGS[2]}" \
  --max-tokens "${SETTINGS[2]}" \
  --tensor-parallel-size "${SETTINGS[3]}" \
  --timeout "${SETTINGS[6]}" \
  "${MODEL_ARGS[@]}"

touch "${OUTPUT_DIR}/COMPLETE"
