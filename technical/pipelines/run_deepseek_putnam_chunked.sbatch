#!/usr/bin/env bash
# Run one DeepSeek Putnam chunk/stage. Submit checkpoints in order: 1, 8, 16, 32.
set -euo pipefail

if (( $# != 12 )); then
  echo "usage: $0 MODEL_CONFIG RUN_CONFIG INPUT_PATH OUTPUT_DIR MODEL_PATH WORKSPACE REPL_COMMAND CHUNKS CHUNK_INDEX SAMPLES GENERATION_OFFSET TARGET_PASS" >&2
  exit 2
fi

MODEL_CONFIG=$1
RUN_CONFIG=$2
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
RUN_MANIFEST=${OUTPUT_DIR}/run_manifest.json

CHUNK_ROOT=${OUTPUT_DIR}/chunks/chunk_${CHUNK_INDEX}
STAGE_DIR=${CHUNK_ROOT}/stage_offset_${GENERATION_OFFSET}_add_${SAMPLES}
CHUNK_INPUT=${CHUNK_ROOT}/input.jsonl
STAGE_MARKER=${STAGE_DIR}/COMPLETE
CUMULATIVE_DIR=${CHUNK_ROOT}/cumulative_pass${TARGET_PASS}

mkdir -p "${OUTPUT_DIR}"
"${PYTHON_BIN}" -m technical.src.run_manifest ensure \
  --manifest "${RUN_MANIFEST}" \
  --model-config "${MODEL_CONFIG}" \
  --run-config "${RUN_CONFIG}" \
  --input "${INPUT_PATH}" \
  --pipeline deepseek-putnam \
  --chunks "${CHUNKS}"
mkdir -p "${STAGE_DIR}"

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
if run["generation"]["sample_schedule"] != [1, 7, 8, 16]:
    raise ValueError("DeepSeek Putnam run profile must use [1, 7, 8, 16]")
if run["assembly"]["mode"] != "standard":
    raise ValueError("DeepSeek Putnam run profile must use standard assembly")
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
print(run["assembly"]["mode"])
print(run["lean"]["profile"])
' "${MODEL_CONFIG}" "${RUN_CONFIG}")

SEED=${SETTINGS[0]}
TEMPERATURE=${SETTINGS[1]}
TOP_P=${SETTINGS[2]}
DTYPE=${SETTINGS[3]}
GPU_MEMORY_UTILIZATION=${SETTINGS[4]}
MAX_MODEL_LEN=${SETTINGS[5]}
MAX_TOKENS=${SETTINGS[6]}
TENSOR_PARALLEL_SIZE=${SETTINGS[7]}
TRUST_REMOTE_CODE=${SETTINGS[8]}
REVISION=${SETTINGS[9]}
PROOF_TIMEOUT=${SETTINGS[10]}
ASSEMBLY_MODE=${SETTINGS[11]}
LEAN_PROFILE=${SETTINGS[12]}

case "${TARGET_PASS}:${GENERATION_OFFSET}:${SAMPLES}" in
  1:0:1) STAGE_SPECS=("0:1") ;;
  8:1:7) STAGE_SPECS=("0:1" "1:7") ;;
  16:8:8) STAGE_SPECS=("0:1" "1:7" "8:8") ;;
  32:16:16) STAGE_SPECS=("0:1" "1:7" "8:8" "16:16") ;;
  *) echo "unsupported checkpoint; use Pass/offset/add 1/0/1, 8/1/7, 16/8/8, or 32/16/16" >&2; exit 2 ;;
esac

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

"${PYTHON_BIN}" -m technical.src.benchmarks.putnam_chunking split \
  --input "${INPUT_PATH}" \
  --output "${CHUNK_INPUT}" \
  --chunks "${CHUNKS}" \
  --index "${CHUNK_INDEX}"

if [[ -f "${STAGE_MARKER}" ]]; then
  "${PYTHON_BIN}" -m technical.src.run_manifest check-marker \
    --manifest "${RUN_MANIFEST}" \
    --marker "${STAGE_MARKER}" \
    --binding kind=putnam-stage \
    --binding chunk_index="${CHUNK_INDEX}" \
    --binding generation_offset="${GENERATION_OFFSET}" \
    --binding samples="${SAMPLES}"
else
  "${PYTHON_BIN}" -m technical.src.generation.generate \
    --input "${CHUNK_INPUT}" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${STAGE_DIR}" \
    --adapter deepseek \
    --assembly-mode "${ASSEMBLY_MODE}" \
    --samples "${SAMPLES}" \
    --generation-offset "${GENERATION_OFFSET}" \
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
    --input "${STAGE_DIR}/to_inference_codes.json" \
    --output "${STAGE_DIR}/code_compilation_full_header.json" \
    --workspace "${WORKSPACE}" \
    --repl-command "${REPL_COMMAND}" \
    --timeout "${PROOF_TIMEOUT}"
  "${PYTHON_BIN}" -m technical.src.run_manifest write-marker \
    --manifest "${RUN_MANIFEST}" \
    --marker "${STAGE_MARKER}" \
    --binding kind=putnam-stage \
    --binding chunk_index="${CHUNK_INDEX}" \
    --binding generation_offset="${GENERATION_OFFSET}" \
    --binding samples="${SAMPLES}"
fi

CUMULATIVE_ARGS=()
for specification in "${STAGE_SPECS[@]}"; do
  offset=${specification%%:*}
  added=${specification##*:}
  required=${CHUNK_ROOT}/stage_offset_${offset}_add_${added}
  "${PYTHON_BIN}" -m technical.src.run_manifest check-marker \
    --manifest "${RUN_MANIFEST}" \
    --marker "${required}/COMPLETE" \
    --binding kind=putnam-stage \
    --binding chunk_index="${CHUNK_INDEX}" \
    --binding generation_offset="${offset}" \
    --binding samples="${added}"
  CUMULATIVE_ARGS+=(--full "${required}/full_records.json")
  CUMULATIVE_ARGS+=(--inference "${required}/to_inference_codes.json")
  CUMULATIVE_ARGS+=(--verification "${required}/code_compilation_full_header.json")
done

if [[ -f "${CUMULATIVE_DIR}/COMPLETE" ]]; then
  "${PYTHON_BIN}" -m technical.src.run_manifest check-marker \
    --manifest "${RUN_MANIFEST}" \
    --marker "${CUMULATIVE_DIR}/COMPLETE" \
    --binding kind=putnam-cumulative \
    --binding chunk_index="${CHUNK_INDEX}" \
    --binding target_pass="${TARGET_PASS}"
else
  "${PYTHON_BIN}" -m technical.src.benchmarks.putnam_chunking cumulative \
    "${CUMULATIVE_ARGS[@]}" \
    --target-pass "${TARGET_PASS}" \
    --output-dir "${CUMULATIVE_DIR}"
  "${PYTHON_BIN}" -m technical.src.run_manifest write-marker \
    --manifest "${RUN_MANIFEST}" \
    --marker "${CUMULATIVE_DIR}/COMPLETE" \
    --binding kind=putnam-cumulative \
    --binding chunk_index="${CHUNK_INDEX}" \
    --binding target_pass="${TARGET_PASS}"
fi

"${PYTHON_BIN}" -m technical.src.benchmarks.putnam_chunking finalize \
  --output-dir "${OUTPUT_DIR}" \
  --total-chunks "${CHUNKS}" \
  --target-pass "${TARGET_PASS}" \
  --run-manifest "${RUN_MANIFEST}"
