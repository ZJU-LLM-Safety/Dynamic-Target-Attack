#!/bin/bash
# ============================================================================
#  Batch runner: Experiment F — Seed Sensitivity Analysis
#
#  5 models × 2 datasets × 5 seeds (per run)
#  Total: 10 experiment runs, each run internally loops over 5 seeds
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${DTA_LOG_DIR:-${PROJECT_ROOT}/logs}"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

# ---- Configuration ----
MODELS=("Llama3" "Qwen2.5" "Mistralv0.3" "Gemma7b" "Vicunav1.5")
DATASETS=(
    "${PROJECT_ROOT}/data/raw/advbench_100.csv"
    "${PROJECT_ROOT}/data/raw/harmBench_100.csv"
)
DATASET_TAGS=("advbench" "harmBench")

SEEDS="42 123 456 789 1024"

# GPU allocation
LOCAL_LLM_DEVICE=2
REF_LLM_DEVICE=1
JUDGE_LLM_DEVICE=2

# Shared hyperparameters (aligned with main experiments)
COMMON_ARGS=(
    --dtype bfloat16
    --local-llm-device ${LOCAL_LLM_DEVICE}
    --ref-local-llm-device ${REF_LLM_DEVICE}
    --judge-llm-device ${JUDGE_LLM_DEVICE}
    --sample-count 30
    --ref-temperature 2.0
    --num-iters 20
    --num-inner-iters 10
    --forward-response-length 20
    --mask-rejection-words
    --start-index 0 --end-index 100
)

# ---- Main loop ----
TOTAL=$((${#MODELS[@]} * ${#DATASETS[@]}))
COUNT=0

for MODEL in "${MODELS[@]}"; do
    for i in "${!DATASETS[@]}"; do
        DATA_PATH="${DATASETS[$i]}"
        TAG="${DATASET_TAGS[$i]}"
        COUNT=$((COUNT + 1))

        VERSION="expF_${MODEL}_${TAG}"

        echo "================================================================="
        echo " [${COUNT}/${TOTAL}] Model=${MODEL}  Dataset=${TAG}  Seeds=${SEEDS}"
        echo " Version: ${VERSION}"
        echo "================================================================="

        python experiments_ablation.py \
            --experiment F \
            --target-llm "${MODEL}" \
            --data-path "${DATA_PATH}" \
            --seeds ${SEEDS} \
            --version "${VERSION}" \
            "${COMMON_ARGS[@]}" \
            2>&1 | tee -a "${LOG_DIR}/expF_${MODEL}_${TAG}.log"

        echo ""
        echo "[✓] Finished ${MODEL} × ${TAG}"
        echo ""
    done
done

echo "================================================================="
echo " All Experiment F runs completed!"
echo " Results: ${PROJECT_ROOT}/data/DTA_ablation/experiment_F/"
echo "================================================================="
