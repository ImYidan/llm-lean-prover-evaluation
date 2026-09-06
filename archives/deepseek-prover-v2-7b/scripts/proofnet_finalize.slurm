#!/usr/bin/env bash
#SBATCH --job-name=proofnet_verify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# Conda's compiler activation hooks read variables that may be unset.
# Enable nounset only after the environment has been activated.
set -eo pipefail

source "${ARCHIVE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/common.sh"

: "${RUN_ID:?RUN_ID must be supplied through sbatch --export}"

EXPECTED_PROBLEMS="${EXPECTED_PROBLEMS:-120}"
EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-32}"
SEED="${SEED:-30}"
RUN_ROOT="${OUTPUT_ROOT}/proofnet_${RUN_ID}"
MERGED_PATH="${RUN_ROOT}/inference.jsonl"

python scripts/merge_inference.py \
  --input-root "${RUN_ROOT}" \
  --output-path "${MERGED_PATH}" \
  --expected-problems "${EXPECTED_PROBLEMS}" \
  --expected-samples "${EXPECTED_SAMPLES}"

python -u src/verify.py \
  --input-path "${MERGED_PATH}" \
  --output-path "${RUN_ROOT}/verification.json" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --timeout 300 \
  --seed "${SEED}"

python src/summarize.py \
  --verification-path "${RUN_ROOT}/verification.json" \
  --output-dir "${RUN_ROOT}/summary" \
  --ks 1,8,16,32

echo "Final results: ${RUN_ROOT}"
