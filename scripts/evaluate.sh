#!/bin/bash
# =============================================================================
# evaluate.sh -- Benchmarking suite for the ORPO-trained Qwen2.5-7B model.
#
# Runs lm-evaluation-harness on four benchmark tasks:
#   1. IFEval    -- Instruction following (0-shot)
#   2. GSM8K     -- Grade-school math reasoning (5-shot)
#   3. MATH      -- Advanced math (Minerva, 4-shot)
#   4. MMLU      -- Broad knowledge, 57 subjects (5-shot)
#
# Usage:
#   bash scripts/evaluate.sh                         # Use defaults
#   bash scripts/evaluate.sh /path/to/model results  # Custom model + output
#
# Prerequisites (install in a separate venv to avoid Unsloth conflicts):
#   pip install lm-eval[vllm] langdetect immutabledict
#
# Target: Kaggle T4 x2 or any CUDA-capable environment.
# =============================================================================

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────

# Model path (merged model or LoRA adapter dir)
MODEL_PATH="${1:-/kaggle/working/orpo_merged_model}"

# Output directory for results
OUTPUT_DIR="${2:-/kaggle/working/eval_results}"

# Batch size: "auto" lets lm-eval choose based on available VRAM
BATCH_SIZE="${3:-auto}"

# Number of GPUs to use
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "1")

# ── Environment Setup ────────────────────────────────────────────────────────

echo "============================================================"
echo "  ORPO Model Evaluation Suite"
echo "============================================================"
echo "  Model path   : ${MODEL_PATH}"
echo "  Output dir   : ${OUTPUT_DIR}"
echo "  Batch size   : ${BATCH_SIZE}"
echo "  GPUs found   : ${NUM_GPUS}"
echo "  Date         : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Check if lm_eval is installed
if ! command -v lm_eval &> /dev/null; then
    echo "[WARN] lm_eval not found. Installing..."
    pip install -q lm-eval langdetect immutabledict
fi

# Check if model path exists
if [ ! -d "${MODEL_PATH}" ]; then
    echo "[ERROR] Model path does not exist: ${MODEL_PATH}"
    echo "[INFO]  Make sure you have run training and model merging first."
    exit 1
fi

# ── Helper Function ──────────────────────────────────────────────────────────

run_benchmark() {
    local task_name=$1
    local num_fewshot=$2
    local task_label=$3

    echo ""
    echo "------------------------------------------------------------"
    echo "  Running: ${task_label}"
    echo "  Task: ${task_name} | Few-shot: ${num_fewshot}"
    echo "  Started: $(date '+%H:%M:%S')"
    echo "------------------------------------------------------------"

    local output_file="${OUTPUT_DIR}/${task_name}_results"

    lm_eval \
        --model hf \
        --model_args "pretrained=${MODEL_PATH},dtype=float16,trust_remote_code=True" \
        --tasks "${task_name}" \
        --num_fewshot "${num_fewshot}" \
        --batch_size "${BATCH_SIZE}" \
        --output_path "${output_file}" \
        --apply_chat_template \
        --log_samples \
        2>&1 | tee "${OUTPUT_DIR}/${task_name}.log"

    local exit_code=${PIPESTATUS[0]}

    if [ ${exit_code} -eq 0 ]; then
        echo "[OK] ${task_label} completed successfully."
    else
        echo "[WARN] ${task_label} exited with code ${exit_code}."
        echo "[INFO] Check log: ${OUTPUT_DIR}/${task_name}.log"
    fi

    echo "  Finished: $(date '+%H:%M:%S')"
    echo ""
}

# ── Run Benchmarks ───────────────────────────────────────────────────────────

echo ""
echo "Starting benchmark suite..."
echo ""

# 1. IFEval -- Instruction Following Evaluation (0-shot)
#    Measures the model's ability to follow explicit formatting instructions.
run_benchmark "ifeval" 0 "IFEval (Instruction Following, 0-shot)"

# 2. GSM8K -- Grade School Math (5-shot, Chain-of-Thought)
#    Tests multi-step arithmetic and word problem reasoning.
run_benchmark "gsm8k" 5 "GSM8K (Math Reasoning, 5-shot)"

# 3. MATH -- Minerva MATH (4-shot)
#    Advanced competition-level mathematics problems.
#    Using minerva_math which is the standard task name in lm-eval.
run_benchmark "minerva_math" 4 "MATH (Advanced Math, 4-shot)"

# 4. MMLU -- Massive Multitask Language Understanding (5-shot)
#    57 subjects spanning STEM, humanities, social sciences, and more.
run_benchmark "mmlu" 5 "MMLU (Broad Knowledge, 5-shot)"

# ── Results Summary ──────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  EVALUATION COMPLETE"
echo "============================================================"
echo "  Results saved to: ${OUTPUT_DIR}"
echo "  Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "  Result files:"
for f in "${OUTPUT_DIR}"/*.log; do
    if [ -f "$f" ]; then
        echo "    - $(basename "$f")"
    fi
done
echo ""
echo "  To view detailed results:"
echo "    cat ${OUTPUT_DIR}/ifeval_results/results.json"
echo "    cat ${OUTPUT_DIR}/gsm8k_results/results.json"
echo "    cat ${OUTPUT_DIR}/minerva_math_results/results.json"
echo "    cat ${OUTPUT_DIR}/mmlu_results/results.json"
echo "============================================================"
