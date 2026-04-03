#!/bin/bash
# Benchmark startup timing for the real Pythia distillation path.
#
# Submit:
#   sbatch downstream/distill_family_study/scripts/run_pythia70_pythia14b_startup_benchmark.sbatch

#SBATCH --job-name=py70_startup_bench
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

cd /home/mila/d/drewd/newManylatents/manylatents
mkdir -p outputs/slurm
export PYTHONPATH="/home/mila/d/drewd/newManylatents/manylatents:${PYTHONPATH:-}"

module load miniconda
export PS1=""
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u
conda activate manylatents

export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1

SCRATCH_ROOT=${SCRATCH_ROOT:-/home/mila/d/drewd/scratch}
export HF_HOME="${SCRATCH_ROOT}/hf_home"
export HF_DATASETS_CACHE="${SCRATCH_ROOT}/hf_datasets"
export TRANSFORMERS_CACHE="${SCRATCH_ROOT}/hf_models"
export TMPDIR="${SCRATCH_ROOT}/tmp"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${TMPDIR}"

python - <<'PY'
import importlib.util
import subprocess
import sys
if importlib.util.find_spec("zstandard") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "zstandard"])
PY

GLOBAL_SEED=${GLOBAL_SEED:-42}
PROBE_IDS_PATH=${PROBE_IDS_PATH:-outputs/pipelines/pythia70_teacher14b_lambda05_pilot_9047636/probe_teacher/probe_teacher/probe_ids.json}
BENCH_PROBE_SIZE=${BENCH_PROBE_SIZE:-32}
BENCH_TOKEN_BUDGET=${BENCH_TOKEN_BUDGET:-32000}
BENCH_OUTPUT=${BENCH_OUTPUT:-outputs/benchmarks/pythia70_teacher14b_startup_${SLURM_JOB_ID}.json}

python scripts/benchmark_pythia_distill_startup.py \
  --output "${BENCH_OUTPUT}" \
  --dataset-name json \
  --dataset-path /network/datasets/pile.var/pile_uncopyrighted/pile-uncopyrighted \
  --dataset-config none \
  --teacher-model-name-or-path EleutherAI/pythia-1.4b \
  --teacher-model-revision main \
  --student-model-name-or-path EleutherAI/pythia-70m \
  --student-model-revision main \
  --tokenizer-name EleutherAI/pythia-70m \
  --train-split train \
  --val-split val \
  --test-split test \
  --probe-split train \
  --probe-ids-path "${PROBE_IDS_PATH}" \
  --exclude-probe-from-train \
  --max-length 1024 \
  --probe-size "${BENCH_PROBE_SIZE}" \
  --token-budget "${BENCH_TOKEN_BUDGET}" \
  --micro-batch-size 1 \
  --seed "${GLOBAL_SEED}" \
  --layer-path "transformer.h[-2]" \
  --precision bf16
