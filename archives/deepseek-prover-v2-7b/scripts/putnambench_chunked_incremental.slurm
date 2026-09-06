#!/usr/bin/env bash
# PutnamBench chunked incremental pass@32 pipeline for DeepSeek-Prover-V2.
#
# One SLURM array task runs one contiguous dataset chunk through:
#   pass@1  = generate 1 sample   at generation offset 0
#   pass@8  = generate 7 samples  at generation offset 1
#   pass@16 = generate 8 samples  at generation offset 8
#   pass@32 = generate 16 samples at generation offset 16

#SBATCH --job-name=putnam_dsv2
#SBATCH --array=0-23%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00

set -eo pipefail

set -u
source "${ARCHIVE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/common.sh"

DATA_PATH="${DATA_PATH:-${DATA_ROOT}/putnambench_lean.jsonl}"
DATASET_NAME="${DATASET_NAME:-putnambench_lean}"
RUN_ID="${RUN_ID:-${SLURM_ARRAY_JOB_ID:-manual_$(date -u +%Y%m%dT%H%M%SZ)}}"
CHUNKS="${CHUNKS:-24}"
CHUNK_INDEX="${SLURM_ARRAY_TASK_ID:-${CHUNK_INDEX:-0}}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${OUTPUT_ROOT}/${DATASET_NAME}_deepseekv2_7b_chunked_incpass32_${RUN_ID}}"
CHUNK_INPUT_DIR="${BASE_OUTPUT_DIR}/dataset_chunks"

SEED="${SEED:-30}"
# DeepSeek-Prover-V2 uses long chain-of-thought before emitting the final ```lean
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VERIFY_TIMEOUT="${VERIFY_TIMEOUT:-300}"
VERIFY_WORKERS="${VERIFY_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Error: DATA_PATH does not exist: ${DATA_PATH}"
  exit 1
fi
if [[ "${CHUNK_INDEX}" -lt 0 || "${CHUNK_INDEX}" -ge "${CHUNKS}" ]]; then
  echo "Error: CHUNK_INDEX=${CHUNK_INDEX} outside [0, ${CHUNKS})."
  exit 1
fi

LAST_CHUNK=$((CHUNKS - 1))
CHUNK_WIDTH=${#LAST_CHUNK}
if [[ "${CHUNK_WIDTH}" -lt 2 ]]; then
  CHUNK_WIDTH=2
fi
CHUNK_NAME=$(printf "chunk_%0${CHUNK_WIDTH}d" "${CHUNK_INDEX}")
CHUNK_FILE="${CHUNK_INPUT_DIR}/${CHUNK_NAME}.jsonl"
CHUNK_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${CHUNK_NAME}"

jsonl_count() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo 0
    return 0
  fi
  python - "${path}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
count = 0
try:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                json.loads(line)
                count += 1
    print(count)
except Exception:
    print(-1)
PY
}

json_list_count() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo 0
    return 0
  fi
  python - "${path}" <<'PY'
import json
import sys
from pathlib import Path

try:
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(len(data) if isinstance(data, list) else -1)
except Exception:
    print(-1)
PY
}

jsonl_complete() {
  local expected="$1"
  local path="$2"
  local count
  count=$(jsonl_count "${path}")
  if [[ "${count}" -ne "${expected}" ]]; then
    echo "Incomplete JSONL: ${path} has ${count}, expected ${expected}."
    return 1
  fi
  return 0
}

json_list_complete() {
  local expected="$1"
  local path="$2"
  local count
  count=$(json_list_count "${path}")
  if [[ "${count}" -ne "${expected}" ]]; then
    echo "Incomplete JSON list: ${path} has ${count}, expected ${expected}."
    return 1
  fi
  return 0
}

prepare_chunks() {
  mkdir -p "${CHUNK_INPUT_DIR}"
  if [[ -f "${CHUNK_FILE}" ]]; then
    return
  fi

  local lock_dir="${CHUNK_INPUT_DIR}.lock"
  if mkdir "${lock_dir}" 2>/dev/null; then
    cleanup_chunk_lock() {
      rmdir "${lock_dir}" 2>/dev/null || true
    }
    trap cleanup_chunk_lock EXIT
    echo "Preparing ${CHUNKS} PutnamBench chunks in ${CHUNK_INPUT_DIR}"
    python scripts/split_jsonl.py \
      --input "${DATA_PATH}" \
      --output-dir "${CHUNK_INPUT_DIR}" \
      --chunks "${CHUNKS}" \
      --prefix "chunk"
    cleanup_chunk_lock
    trap - EXIT
  else
    echo "Another task is preparing chunks; waiting for ${CHUNK_FILE}"
    while [[ ! -f "${CHUNK_FILE}" ]]; do
      sleep 10
    done
  fi
}

run_stage_if_needed() {
  local target_pass="$1"
  local add_samples="$2"
  local offset="$3"
  local stage_dir="$4"
  local inference_output="${stage_dir}/inference.jsonl"
  local verification_output="${stage_dir}/verification.json"
  local expected_records=$((CHUNK_PROBLEM_COUNT * add_samples))

  mkdir -p "${stage_dir}"
  if ! jsonl_complete "${expected_records}" "${inference_output}"; then
    echo
    echo "----- ${CHUNK_NAME} stage pass@${target_pass}: inference add=${add_samples}, offset=${offset} -----"
    nvidia-smi
    python -u src/inference.py \
      --input-path "${CHUNK_FILE}" \
      --output-dir "${stage_dir}" \
      --model-path "${MODEL_DIR}" \
      --split none \
      --samples "${add_samples}" \
      --generation-offset "${offset}" \
      --tensor-parallel-size 1 \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --temperature "${TEMPERATURE}" \
      --top-p "${TOP_P}" \
      --seed "${SEED}" \
      --batch-size "${BATCH_SIZE}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  else
    echo "Skipping inference for ${stage_dir}; ${expected_records} records already complete."
  fi

  if ! json_list_complete "${expected_records}" "${verification_output}"; then
    echo
    echo "----- ${CHUNK_NAME} stage pass@${target_pass}: header-preserved verification -----"
    python -u src/verify.py \
      --input-path "${inference_output}" \
      --output-path "${verification_output}" \
      --workers "${VERIFY_WORKERS}" \
      --timeout "${VERIFY_TIMEOUT}" \
      --seed "${SEED}"
  else
    echo "Skipping verification for ${stage_dir}; ${expected_records} records already complete."
  fi
}

summarize_cumulative() {
  local target_pass="$1"
  local cumulative_dir="${CHUNK_OUTPUT_DIR}/cumulative_pass@${target_pass}"
  mkdir -p "${cumulative_dir}"

  python scripts/merge_records.py \
    --input-format jsonl \
    --output-format jsonl \
    --output-path "${cumulative_dir}/inference.jsonl" \
    --expected-problems "${CHUNK_PROBLEM_COUNT}" \
    --expected-samples "${target_pass}" \
    "${INFERENCE_FILES[@]}"

  python scripts/merge_records.py \
    --input-format json \
    --output-format json \
    --output-path "${cumulative_dir}/verification.json" \
    --expected-problems "${CHUNK_PROBLEM_COUNT}" \
    --expected-samples "${target_pass}" \
    "${VERIFICATION_FILES[@]}"

  python src/summarize.py \
    --verification-path "${cumulative_dir}/verification.json" \
    --output-dir "${cumulative_dir}/summary" \
    --ks 1,8,16,32

  python scripts/putnam_progress_summary.py \
    --run-root "${BASE_OUTPUT_DIR}" \
    --pass-target "${target_pass}" \
    --chunks "${CHUNKS}"
}

mkdir -p "${BASE_OUTPUT_DIR}" "${CHUNK_OUTPUT_DIR}"
prepare_chunks
CHUNK_PROBLEM_COUNT=$(wc -l < "${CHUNK_FILE}")

echo "========================================================"
echo "DeepSeek PutnamBench Chunked Incremental Pipeline"
echo "========================================================"
echo "Dataset: ${DATASET_NAME} (${DATA_PATH})"
echo "Chunk: ${CHUNK_NAME}/${CHUNKS} (${CHUNK_FILE}, lines=${CHUNK_PROBLEM_COUNT})"
echo "Model: ${MODEL_DIR}"
echo "Stages: pass@1 -> pass@8 -> pass@16 -> pass@32"
echo "GPU: 1 | Verify workers: ${VERIFY_WORKERS} | Verify timeout: ${VERIFY_TIMEOUT}"
echo "Output root: ${BASE_OUTPUT_DIR}"
echo "Chunk output: ${CHUNK_OUTPUT_DIR}"
echo "Run ID: ${RUN_ID}"
echo "========================================================"

PASS_TARGETS=(1 8 16 32)
ADD_SAMPLES=(1 7 8 16)
OFFSETS=(0 1 8 16)

INFERENCE_FILES=()
VERIFICATION_FILES=()

for i in "${!PASS_TARGETS[@]}"; do
  target="${PASS_TARGETS[$i]}"
  add="${ADD_SAMPLES[$i]}"
  offset="${OFFSETS[$i]}"
  stage_dir="${CHUNK_OUTPUT_DIR}/stage_pass@${target}_add${add}_offset${offset}"

  run_stage_if_needed "${target}" "${add}" "${offset}" "${stage_dir}"

  INFERENCE_FILES+=("${stage_dir}/inference.jsonl")
  VERIFICATION_FILES+=("${stage_dir}/verification.json")

  echo
  echo "----- ${CHUNK_NAME}: cumulative summary pass@${target} -----"
  summarize_cumulative "${target}"
done

echo
echo "Completed ${CHUNK_NAME}. Results are in ${CHUNK_OUTPUT_DIR}"
