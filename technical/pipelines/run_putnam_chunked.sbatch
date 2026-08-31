#!/usr/bin/env bash
# Run one chunk/stage. Submit checkpoints in order: Pass@1, 8, 16, then 32.
set -euo pipefail

if (( $# != 12 )); then
  echo "usage: $0 MODEL_CONFIG BENCHMARK_CONFIG INPUT_PATH OUTPUT_DIR MODEL_PATH WORKSPACE REPL_COMMAND CHUNKS CHUNK_INDEX SAMPLES GENERATION_OFFSET TARGET_PASS" >&2
  exit 2
fi

MODEL_CONFIG=$1
BENCHMARK_CONFIG=$2
INPUT_PATH=$3
OUTPUT_DIR=$4
MODEL_PATH=$5
WORKSPACE=$6
REPL_COMMAND=$7
CHUNKS=$8
CHUNK_INDEX=$9
SAMPLES=${10}
GENERATION_OFFSET=${11}
TARGET_PASS=${12}
PYTHON_BIN=${PYTHON_BIN:-python}

readarray -t SETTINGS < <("${PYTHON_BIN}" -c '
import sys
from pathlib import Path
from technical.src.config import load_config
model = load_config(Path(sys.argv[1]))
benchmark = load_config(Path(sys.argv[2]))
if benchmark["verification"]["mode"] != "full_header_repl":
    raise ValueError("benchmark verification.mode must be full_header_repl")
print(model["generation"]["temperature"])
print(model["generation"]["top_p"])
print(model["generation"]["max_model_len"])
print(model["model"]["tensor_parallel_size"])
print(str(model["model"]["trust_remote_code"]).lower())
print(model["model"]["revision"] or "__NONE__")
print(benchmark["verification"]["timeout"])
' "${MODEL_CONFIG}" "${BENCHMARK_CONFIG}")

case "${TARGET_PASS}:${GENERATION_OFFSET}:${SAMPLES}" in
  1:0:1) STAGE_SPECS=("0:1") ;;
  8:1:7) STAGE_SPECS=("0:1" "1:7") ;;
  16:8:8) STAGE_SPECS=("0:1" "1:7" "8:8") ;;
  32:16:16) STAGE_SPECS=("0:1" "1:7" "8:8" "16:16") ;;
  *) echo "unsupported checkpoint; use Pass/offset/add 1/0/1, 8/1/7, 16/8/8, or 32/16/16" >&2; exit 2 ;;
esac

MODEL_ARGS=()
MAX_TOKENS=$((SETTINGS[2] - 1))
if [[ "${SETTINGS[4]}" == "true" ]]; then
  MODEL_ARGS+=(--trust-remote-code)
fi
if [[ "${SETTINGS[5]}" != "__NONE__" ]]; then
  MODEL_ARGS+=(--revision "${SETTINGS[5]}")
fi

CHUNK_ROOT=${OUTPUT_DIR}/chunks/chunk_${CHUNK_INDEX}
STAGE_DIR=${CHUNK_ROOT}/stage_offset_${GENERATION_OFFSET}_add_${SAMPLES}
CHUNK_INPUT=${CHUNK_ROOT}/input.jsonl
STAGE_MARKER=${STAGE_DIR}/COMPLETE
CUMULATIVE_DIR=${CHUNK_ROOT}/cumulative_pass${TARGET_PASS}

mkdir -p "${OUTPUT_DIR}" "${STAGE_DIR}"
rm -f "${STAGE_MARKER}"
"${PYTHON_BIN}" -m technical.src.benchmarks.putnam_chunking split \
  --input "${INPUT_PATH}" --output "${CHUNK_INPUT}" \
  --chunks "${CHUNKS}" --index "${CHUNK_INDEX}"

"${PYTHON_BIN}" -m technical.src.generation.generate \
  --input "${CHUNK_INPUT}" --model-path "${MODEL_PATH}" --output-dir "${STAGE_DIR}" \
  --samples "${SAMPLES}" --generation-offset "${GENERATION_OFFSET}" \
  --temperature "${SETTINGS[0]}" --top-p "${SETTINGS[1]}" \
  --max-model-len "${SETTINGS[2]}" --max-tokens "${MAX_TOKENS}" \
  --tensor-parallel-size "${SETTINGS[3]}" "${MODEL_ARGS[@]}"

"${PYTHON_BIN}" -m technical.src.verification.verify_full_header_repl \
  --input "${STAGE_DIR}/to_inference_codes.json" \
  --output "${STAGE_DIR}/code_compilation_full_header.json" \
  --workspace "${WORKSPACE}" --repl-command "${REPL_COMMAND}" \
  --timeout "${SETTINGS[6]}"
touch "${STAGE_MARKER}"

CUMULATIVE_ARGS=()
for specification in "${STAGE_SPECS[@]}"; do
  offset=${specification%%:*}
  added=${specification##*:}
  required=${CHUNK_ROOT}/stage_offset_${offset}_add_${added}
  if [[ ! -f "${required}/COMPLETE" ]]; then
    echo "missing completed prerequisite stage: ${required}" >&2
    exit 1
  fi
  CUMULATIVE_ARGS+=(--full "${required}/full_records.json")
  CUMULATIVE_ARGS+=(--inference "${required}/to_inference_codes.json")
  CUMULATIVE_ARGS+=(--verification "${required}/code_compilation_full_header.json")
done
"${PYTHON_BIN}" -m technical.src.benchmarks.putnam_chunking cumulative \
  "${CUMULATIVE_ARGS[@]}" --target-pass "${TARGET_PASS}" --output-dir "${CUMULATIVE_DIR}"
touch "${CUMULATIVE_DIR}/COMPLETE"

PROGRESS_ARGS=()
FULL_MERGE_ARGS=()
INFERENCE_MERGE_ARGS=()
VERIFICATION_MERGE_ARGS=()
for (( index=0; index<CHUNKS; index++ )); do
  candidate_dir=${OUTPUT_DIR}/chunks/chunk_${index}/cumulative_pass${TARGET_PASS}
  candidate_marker=${candidate_dir}/COMPLETE
  if [[ -f "${candidate_marker}" ]]; then
    PROGRESS_ARGS+=(--chunk "${index}=${candidate_dir}/code_compilation_full_header.json")
    FULL_MERGE_ARGS+=(--input "${candidate_dir}/full_records.json")
    INFERENCE_MERGE_ARGS+=(--input "${candidate_dir}/to_inference_codes.json")
    VERIFICATION_MERGE_ARGS+=(--input "${candidate_dir}/code_compilation_full_header.json")
  fi
done

GLOBAL_DIR=${OUTPUT_DIR}/cumulative_pass${TARGET_PASS}
"${PYTHON_BIN}" -m technical.src.benchmarks.putnam_chunking merge \
  "${FULL_MERGE_ARGS[@]}" --output "${GLOBAL_DIR}/full_records.json"
"${PYTHON_BIN}" -m technical.src.benchmarks.putnam_chunking merge \
  "${INFERENCE_MERGE_ARGS[@]}" --output "${GLOBAL_DIR}/to_inference_codes.json"
"${PYTHON_BIN}" -m technical.src.benchmarks.putnam_chunking merge \
  "${VERIFICATION_MERGE_ARGS[@]}" --output "${GLOBAL_DIR}/code_compilation_full_header.json"
"${PYTHON_BIN}" -m technical.src.benchmarks.putnam_chunking progress \
  "${PROGRESS_ARGS[@]}" --total-chunks "${CHUNKS}" --target-pass "${TARGET_PASS}" \
  --output "${OUTPUT_DIR}/progress_pass${TARGET_PASS}.json"
