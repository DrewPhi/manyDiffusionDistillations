#!/bin/bash
# Real-config smoke test for the Pythia distillation sweep pipeline.
#
# Goal:
# - exercise the exact clean-split config and probe-ID contract used by the full job
# - keep runtime short enough to fail fast if pipeline wiring is still broken
#
# Submit:
#   sbatch downstream/distill_family_study/scripts/run_pythia70_pythia14b_loss_logging_smoke.sbatch

#SBATCH --job-name=py70_loss_smoke
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

cd /home/mila/d/drewd/newManylatents/manylatents
mkdir -p outputs/slurm
DOWNSTREAM_CONFIG_DIR="$(pwd)/downstream/distill_family_study/configs"

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
DATA_ORDER_SEED=${DATA_ORDER_SEED:-42}
PROBE_IDS_PATH=${PROBE_IDS_PATH:-outputs/pipelines/pythia70_teacher14b_loss_logging_sweep_9101271/probe_teacher/probe_teacher/probe_ids.json}
SMOKE_PROBE_SIZE=${SMOKE_PROBE_SIZE:-32}
SMOKE_TOKEN_BUDGET=${SMOKE_TOKEN_BUDGET:-8192}
SMOKE_MAX_STEPS=${SMOKE_MAX_STEPS:-1}

python -m manylatents.main \
  "hydra.searchpath=[file://${DOWNSTREAM_CONFIG_DIR}]" \
  experiment=pythia_distill_sweep_template \
  name=pythia70_teacher14b_loss_logging_smoke_${SLURM_JOB_ID} \
  output_dir=./outputs \
  stage_pipeline.resume=false \
  stage_pipeline.params.teacher.model_name_or_path=EleutherAI/pythia-1.4b \
  stage_pipeline.params.teacher.model_revision=main \
  'stage_pipeline.params.teacher.penultimate_layer_spec="transformer.h[-2]"' \
  stage_pipeline.params.student.model_name_or_path=EleutherAI/pythia-70m \
  stage_pipeline.params.student.model_revision=main \
  stage_pipeline.params.student.init_from_scratch=true \
  stage_pipeline.params.student.penultimate_dim=512 \
  'stage_pipeline.params.student.penultimate_layer_spec="transformer.h[-2]"' \
  stage_pipeline.params.data.dataset_name=json \
  stage_pipeline.params.data.dataset_config=null \
  stage_pipeline.params.data.dataset_path=/network/datasets/pile.var/pile_uncopyrighted/pile-uncopyrighted \
  stage_pipeline.params.data.dataset_revision=null \
  stage_pipeline.params.data.text_field=text \
  stage_pipeline.params.data.tokenizer_name=EleutherAI/pythia-70m \
  stage_pipeline.params.data.train_split=train \
  stage_pipeline.params.data.val_split=val \
  stage_pipeline.params.data.test_split=test \
  stage_pipeline.params.data.probe_split=train \
  stage_pipeline.params.data.exclude_probe_from_train=true \
  stage_pipeline.params.data.max_length=1024 \
  "stage_pipeline.params.data.token_budget=${SMOKE_TOKEN_BUDGET}" \
  stage_pipeline.params.data.train_example_offset=0 \
  stage_pipeline.params.probe.sampling=random \
  "stage_pipeline.params.probe.seed=${GLOBAL_SEED}" \
  "stage_pipeline.params.probe.ids_path=${PROBE_IDS_PATH}" \
  "stage_pipeline.params.probe.size=${SMOKE_PROBE_SIZE}" \
  stage_pipeline.params.probe.train_fraction=0.8 \
  stage_pipeline.params.probe.eval_fraction=0.2 \
  stage_pipeline.params.probe.save_ids=false \
  stage_pipeline.params.diffusion.knn=35 \
  stage_pipeline.params.diffusion.symmetric=false \
  stage_pipeline.params.diffusion.metric=euclidean \
  stage_pipeline.params.phate.n_components=512 \
  stage_pipeline.params.phate.knn=35 \
  stage_pipeline.params.phate.t=auto \
  stage_pipeline.params.phate.decay=40 \
  stage_pipeline.params.phate.gamma=1.0 \
  "stage_pipeline.params.phate.random_state=${GLOBAL_SEED}" \
  stage_pipeline.params.phate.n_pca=null \
  stage_pipeline.params.phate.n_landmark=null \
  stage_pipeline.params.phate.n_jobs=-1 \
  stage_pipeline.params.phate.verbose=false \
  stage_pipeline.params.phate.fit_fraction=1.0 \
  stage_pipeline.params.training.precision=bf16 \
  stage_pipeline.params.training.global_batch_size=8 \
  stage_pipeline.params.training.micro_batch_size=1 \
  stage_pipeline.params.training.grad_accum_steps=8 \
  "stage_pipeline.params.training.max_steps=${SMOKE_MAX_STEPS}" \
  stage_pipeline.params.training.eval_every_n_steps=1 \
  stage_pipeline.params.training.save_every_n_steps=1 \
  stage_pipeline.params.training.save_top_k=1 \
  stage_pipeline.params.training.gradient_clip_norm=1.0 \
  stage_pipeline.params.training.train_history_every_n_steps=1 \
  stage_pipeline.params.training.lm_loss_weight=1.0 \
  stage_pipeline.params.optimizer.name=adamw \
  stage_pipeline.params.optimizer.learning_rate=3.0e-4 \
  'stage_pipeline.params.optimizer.betas=[0.9,0.95]' \
  stage_pipeline.params.optimizer.eps=1.0e-8 \
  stage_pipeline.params.optimizer.weight_decay=0.1 \
  stage_pipeline.params.lr_scheduler.type=cosine \
  stage_pipeline.params.lr_scheduler.warmup_steps=1 \
  stage_pipeline.params.lr_scheduler.min_lr=3.0e-5 \
  stage_pipeline.params.regularization.dropout=0.0 \
  stage_pipeline.params.regularization.label_smoothing=0.0 \
  stage_pipeline.params.alignment.loss=mse \
  stage_pipeline.params.alignment.mse_reduction=mean \
  stage_pipeline.params.alignment.lambda_schedule=constant \
  stage_pipeline.params.alignment.lambda_ramp_steps=0 \
  stage_pipeline.params.alignment.batch_size=4 \
  stage_pipeline.params.alignment.sample_every_n_steps=1 \
  "stage_pipeline.params.seeds.global_seed=${GLOBAL_SEED}" \
  stage_pipeline.params.seeds.model_init_seed=null \
  "stage_pipeline.params.seeds.data_order_seed=${DATA_ORDER_SEED}" \
  stage_pipeline.params.seeds.dataloader_seed=null \
  "stage_pipeline.params.seeds.phate_seed=${GLOBAL_SEED}" \
  'stage_pipeline.params.sweep.teacher_model=[EleutherAI/pythia-1.4b]' \
  'stage_pipeline.params.sweep.student_model=[EleutherAI/pythia-70m]' \
  "stage_pipeline.params.sweep.seed=[${GLOBAL_SEED}]" \
  'stage_pipeline.params.sweep.lambda_align=[0.3]' \
  'stage_pipeline.params.sweep.learning_rate=[3.0e-4]' \
  'stage_pipeline.params.sweep.max_length=[1024]' \
  "stage_pipeline.params.sweep.token_budget=[${SMOKE_TOKEN_BUDGET}]" \
  'stage_pipeline.params.sweep.train_example_offset=[0]'
