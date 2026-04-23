#!/bin/bash
# Tamia H100 smoke for the distillation algo module.
#
# Full-node H100 allocation (tamia policy). Only uses 1 GPU for the smoke;
# the other 3 are idle - acceptable for a ~15 min smoke. If this ever grows
# into a real multi-step training, shard across all 4.
#
# Submit from the manyDiff-algo checkout:
#   sbatch examples/distillation/run_smoke.sbatch

#SBATCH --job-name=distill_smoke
#SBATCH --account=aip-wolfg
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_smoke_%j.out
#SBATCH --error=slurm_smoke_%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/c/cesarmvc/manyDiff-algo}
VENV=${VENV:-/scratch/c/cesarmvc/manyDiffusionDistillations/.venv}
PILE_MINI=${PILE_MINI:-/scratch/c/cesarmvc/pile_mini}
OUT_DIR=${OUT_DIR:-${REPO_ROOT}/smoke_out/${SLURM_JOB_ID:-dev}}

cd "${REPO_ROOT}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/scratch/c/cesarmvc/hf_home
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[sbatch] REPO_ROOT=${REPO_ROOT}"
echo "[sbatch] VENV=${VENV}"
echo "[sbatch] PILE_MINI=${PILE_MINI}"
echo "[sbatch] OUT_DIR=${OUT_DIR}"
echo "[sbatch] job_id=${SLURM_JOB_ID:-dev} host=$(hostname)"

nvidia-smi || true

"${VENV}/bin/python" examples/distillation/run_smoke.py \
    --pile-mini "${PILE_MINI}" \
    --out-dir "${OUT_DIR}"
