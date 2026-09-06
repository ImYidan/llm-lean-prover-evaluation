#!/usr/bin/env bash
#SBATCH --job-name=proofnet_gen
#SBATCH --array=0-3%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Conda's compiler activation hooks read variables that may be unset.
# Enable nounset only after the environment has been activated.
set -eo pipefail

source "${ARCHIVE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/common.sh"

: "${RUN_ID:?RUN_ID must be supplied through sbatch --export}"

INPUT_PATH="${INPUT_PATH:-${DATA_ROOT}/proofnet.jsonl}"
SAMPLES="${SAMPLES:-32}"
SEED="${SEED:-30}"
NUM_SHARDS="${NUM_SHARDS:-4}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID}"
RUN_ROOT="${OUTPUT_ROOT}/proofnet_${RUN_ID}"
SHARD_DIR="${RUN_ROOT}/shard_${SHARD_INDEX}"
mkdir -p "${SHARD_DIR}"

limit_args=()
if [[ -n "${LIMIT:-}" ]]; then
  limit_args=(--limit "${LIMIT}")
fi

echo "run_id=${RUN_ID}"
echo "job_id=${SLURM_JOB_ID}"
echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
echo "input_path=${INPUT_PATH}"
echo "shard=${SHARD_INDEX}/${NUM_SHARDS}"
echo "samples=${SAMPLES}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

python -u src/inference.py \
  --input-path "${INPUT_PATH}" \
  --output-dir "${SHARD_DIR}" \
  --model-path "${MODEL_DIR}" \
  --samples "${SAMPLES}" \
  --tensor-parallel-size 1 \
  --max-model-len 40960 \
  --max-new-tokens 32768 \
  --temperature 1.0 \
  --top-p 0.95 \
  --seed "${SEED}" \
  --batch-size 1 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.90 \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${SHARD_INDEX}" \
  "${limit_args[@]}"
