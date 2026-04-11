#!/bin/bash
# Generic single-run wrapper for one materialized family-study spec.

#SBATCH --job-name=distill_family
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100l:1

set -euo pipefail

if [[ -n "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="${REPO_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
fi
cd "${REPO_ROOT}"
mkdir -p outputs/slurm

if [[ -z "${RUN_SPEC_PATH:-}" ]]; then
  echo "ERROR: RUN_SPEC_PATH must point to a materialized run-spec JSON file."
  exit 2
fi

module load miniconda
export PS1=""
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u
conda activate manylatents

export WANDB_ENTITY=${WANDB_ENTITY:-cesar-valdez-mcgill-university}
export WANDB_PROJECT=${WANDB_PROJECT:-diffusion-distillation}
export WANDB_MODE=${WANDB_MODE:-online}
if [[ "${WANDB_MODE}" != "disabled" ]]; then
  if [[ -z "${WANDB_ENTITY:-}" ]]; then
    echo "ERROR: WANDB_ENTITY is not set. Export your W&B team/entity before submitting."
    exit 2
  fi
fi

export HYDRA_FULL_ERROR=1

SCRATCH_ROOT=${SCRATCH_ROOT:-${SCRATCH:-${HOME}/scratch}}
export HF_HOME="${SCRATCH_ROOT}/hf_home"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${SCRATCH_ROOT}/hf_datasets"
export TRANSFORMERS_CACHE="${SCRATCH_ROOT}/hf_models"
export TMPDIR="${SCRATCH_ROOT}/tmp"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${TMPDIR}"
for CACHE_DIR in "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${TMPDIR}"; do
  mkdir -p "${CACHE_DIR}"
  if [[ ! -w "${CACHE_DIR}" ]]; then
    echo "ERROR: cache path is not writable: ${CACHE_DIR}"
    exit 2
  fi
done
echo "[hf-cache] HF_HOME=${HF_HOME}"
echo "[hf-cache] HF_HUB_CACHE=${HF_HUB_CACHE}"
echo "[hf-cache] HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
echo "[hf-cache] TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}"

python - <<'PY'
import importlib.util
import subprocess
import sys
if importlib.util.find_spec("zstandard") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "zstandard"])
PY

PREPARE_HF_CACHE=${PREPARE_HF_CACHE:-1}
if [[ "${PREPARE_HF_CACHE}" == "1" ]]; then
  python downstream/distill_family_study/scripts/prefetch_family_hf_assets.py --run-spec "${RUN_SPEC_PATH}"
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python downstream/distill_family_study/scripts/run_materialized_distill_run.py --run-spec "${RUN_SPEC_PATH}"
