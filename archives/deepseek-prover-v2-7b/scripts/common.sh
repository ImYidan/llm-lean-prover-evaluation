#!/usr/bin/env bash
# Activate the historical Python environment before invoking any batch entry.
set -euo pipefail

ARCHIVE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PROJECT_ROOT="${ARCHIVE_ROOT}"
: "${DATA_ROOT:?Set DATA_ROOT to the prepared benchmark input directory}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a writable run directory}"
: "${MODEL_DIR:?Set MODEL_DIR to the pinned local model snapshot}"
: "${DEEPSEEK_LEAN_WORKSPACE:?Set DEEPSEEK_LEAN_WORKSPACE to the matching Lean profile}"
: "${DEEPSEEK_LAKE_PATH:?Set DEEPSEEK_LAKE_PATH to the Lake executable}"
: "${DEEPSEEK_REPL_PATH:?Set DEEPSEEK_REPL_PATH to the matching REPL executable}"
export DATA_ROOT OUTPUT_ROOT MODEL_DIR
export DEEPSEEK_LEAN_WORKSPACE DEEPSEEK_LAKE_PATH DEEPSEEK_REPL_PATH

export HF_HOME="${HF_HOME:-${OUTPUT_ROOT}/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${OUTPUT_ROOT}/cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${OUTPUT_ROOT}/cache/triton}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${OUTPUT_ROOT}/cache/vllm}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
mkdir -p "${OUTPUT_ROOT}" "${HF_HOME}" "${TORCH_HOME}" "${TRITON_CACHE_DIR}" "${VLLM_CACHE_ROOT}"
cd "${PROJECT_ROOT}"
