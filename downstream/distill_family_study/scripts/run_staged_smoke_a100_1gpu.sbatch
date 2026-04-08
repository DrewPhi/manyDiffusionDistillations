#!/bin/bash
# Minimal staged-training smoke on one A100 GPU.

#SBATCH --job-name=staged_smoke_a100
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100l:1

set -euo pipefail

REPO_ROOT=/home/mila/d/drewd/codeReview/manyDiffusionDistillations
cd "${REPO_ROOT}"
mkdir -p outputs/slurm
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

module load miniconda
export PS1=""
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u
conda activate manylatents

export WANDB_ENTITY=${WANDB_ENTITY:-cesar-valdez-mcgill-university}
export WANDB_PROJECT=${WANDB_PROJECT:-diffusion-distillation}
export WANDB_MODE=${WANDB_MODE:-online}
export HYDRA_FULL_ERROR=1

SCRATCH_ROOT=${SCRATCH_ROOT:-/home/mila/d/drewd/scratch}
export HF_HOME="${SCRATCH_ROOT}/hf_home"
export HF_DATASETS_CACHE="${SCRATCH_ROOT}/hf_datasets"
export TRANSFORMERS_CACHE="${SCRATCH_ROOT}/hf_models"
export TMPDIR="${SCRATCH_ROOT}/tmp"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${TMPDIR}"

MANIFEST_DIR=${MANIFEST_DIR:-results/study_manifests/staged_smoke_a100_1gpu}
RUN_SPECS_DIR="${MANIFEST_DIR}/run_specs"
mkdir -p "${RUN_SPECS_DIR}"
find "${RUN_SPECS_DIR}" -maxdepth 1 -name '*.json' -delete

python downstream/distill_family_study/scripts/submit_distill_study.py \
  --study-config downstream/distill_family_study/configs/study/staged_smoke_a100_1gpu.yaml \
  --manifest-dir "${MANIFEST_DIR}"

mapfile -t RUN_SPEC_PATHS < <(find "${RUN_SPECS_DIR}" -maxdepth 1 -name '*.json' | sort)
if [[ "${#RUN_SPEC_PATHS[@]}" -eq 0 ]]; then
  echo "ERROR: no run spec materialized under ${RUN_SPECS_DIR}"
  exit 2
fi

for RUN_SPEC_PATH in "${RUN_SPEC_PATHS[@]}"; do
  RUN_NAME=$(python - <<'PY' "${RUN_SPEC_PATH}"
import json, sys
payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(payload["run_name"])
PY
  )
  rm -rf "outputs/pipelines/${RUN_NAME}"
  python downstream/distill_family_study/scripts/prefetch_family_hf_assets.py --run-spec "${RUN_SPEC_PATH}"
done

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for RUN_SPEC_PATH in "${RUN_SPEC_PATHS[@]}"; do
  RUN_NAME=$(python - <<'PY' "${RUN_SPEC_PATH}"
import json, sys
payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(payload["run_name"])
PY
  )

  python downstream/distill_family_study/scripts/run_materialized_distill_run.py --run-spec "${RUN_SPEC_PATH}"

  QUEUE_PATH=$(find "outputs/pipelines/${RUN_NAME}" -path '*/analysis_queue.jsonl' | sort | head -n 1 || true)
  if [[ -z "${QUEUE_PATH}" || ! -f "${QUEUE_PATH}" ]]; then
    echo "ERROR: expected analysis queue at ${QUEUE_PATH}"
    exit 2
  fi

  python downstream/distill_family_study/scripts/process_analysis_queue.py \
    --queue-path "${QUEUE_PATH}"
done

python downstream/distill_family_study/scripts/aggregate_distill_study.py \
  --manifest-dir "${MANIFEST_DIR}" \
  --output-dir "results/publication_within_family/staged_smoke_a100_1gpu"

python downstream/distill_family_study/scripts/plot_distill_study.py \
  --input-dir "results/publication_within_family/staged_smoke_a100_1gpu" \
  --output-dir "results/publication_within_family/staged_smoke_a100_1gpu/figures"
