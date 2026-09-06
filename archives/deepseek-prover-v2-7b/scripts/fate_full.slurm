#!/usr/bin/env bash
#SBATCH --job-name=deepseekv2_fate_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# FATE FULL (DeepSeek-Prover-V2-7B), pass@32, verified against v4.28 Mathlib.
# Parametrized by DATA_PATH + DATASET_NAME (pass via sbatch --export), e.g.:
#   sbatch --export=ALL,DATA_PATH=${DATA_ROOT}/fate_m.jsonl,DATASET_NAME=fate_m scripts/fate_full.slurm
# Params aligned with the other DeepSeek runs: temp 1.0, top_p 0.95, samples 32,
# max_model_len 32768, max_new_tokens 8192, seed 30, tensor-parallel 1.
# summarize emits pass@1/8/16/32.
set -eo pipefail


source "${ARCHIVE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/common.sh"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
SEED="${SEED:-30}"
DATA_PATH="${DATA_PATH:-${DATA_ROOT}/fate_m.jsonl}"
DATASET_NAME="${DATASET_NAME:-fate_m}"
SAMPLES="${SAMPLES:-32}"
RUN_DIR="${OUTPUT_ROOT}/${DATASET_NAME}_deepseekv2_7b_pass32_${RUN_ID}"
mkdir -p "${RUN_DIR}"
echo "FATE FULL: ${DATASET_NAME} (${DATA_PATH}) samples=${SAMPLES} -> ${RUN_DIR}"

nvidia-smi

python -u src/inference.py \
  --input-path "${DATA_PATH}" \
  --output-dir "${RUN_DIR}" \
  --model-path "${MODEL_DIR}" \
  --samples "${SAMPLES}" \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --max-new-tokens 8192 \
  --temperature 1.0 \
  --top-p 0.95 \
  --seed "${SEED}" \
  --batch-size 1 \
  --max-num-seqs 8

python src/verify.py \
  --input-path "${RUN_DIR}/inference.jsonl" \
  --output-path "${RUN_DIR}/verification.json" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --timeout 4000 \
  --seed "${SEED}"

python src/summarize.py \
  --verification-path "${RUN_DIR}/verification.json" \
  --output-dir "${RUN_DIR}/summary"

echo "===================================================="
echo "FATE FULL (${DATASET_NAME}) COMPLETE. Summary: ${RUN_DIR}/summary/summary.json"
cat "${RUN_DIR}/summary/summary.json" || true
echo "===================================================="
