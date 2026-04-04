#!/bin/bash
# Full distillation sweep run:
# teacher=pythia-1.4b, student=pythia-70m, lambda_align in {0.0, 0.5, 1.0}
#
# Submit:
#   sbatch downstream/distill_family_study/scripts/run_pythia70_pythia14b_lambda_sweep.sbatch

#SBATCH --job-name=py70_distill_lam
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --time=24:00:00
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

# Route HF/datasets/temp caches to scratch to avoid home quota pressure.
SCRATCH_ROOT=${SCRATCH_ROOT:-/home/mila/d/drewd/scratch}
export HF_HOME="${SCRATCH_ROOT}/hf_home"
export HF_DATASETS_CACHE="${SCRATCH_ROOT}/hf_datasets"
export TRANSFORMERS_CACHE="${SCRATCH_ROOT}/hf_models"
export TMPDIR="${SCRATCH_ROOT}/tmp"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${TMPDIR}"

# Ensure local .jsonl.zst shards are readable by datasets/json loader.
python - <<'PY'
import importlib.util
import subprocess
import sys
if importlib.util.find_spec("zstandard") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "zstandard"])
PY

python -m manylatents.main \
  "hydra.searchpath=[file://${DOWNSTREAM_CONFIG_DIR}]" \
  experiment=pythia_distill_sweep_template \
  name=pythia70_teacher14b_lambda_sweep_${SLURM_JOB_ID} \
  output_dir=./outputs \
  stage_pipeline.resume=false \
  stage_pipeline.params.teacher.model_name_or_path=EleutherAI/pythia-1.4b \
  stage_pipeline.params.teacher.model_revision=main \
  stage_pipeline.params.teacher.penultimate_layer_spec=transformer.h[-2] \
  stage_pipeline.params.student.model_name_or_path=EleutherAI/pythia-70m \
  stage_pipeline.params.student.model_revision=main \
  stage_pipeline.params.student.init_from_scratch=true \
  stage_pipeline.params.student.penultimate_dim=512 \
  stage_pipeline.params.student.penultimate_layer_spec=transformer.h[-2] \
  stage_pipeline.params.data.dataset_name=json \
  stage_pipeline.params.data.dataset_config=null \
  stage_pipeline.params.data.dataset_path=/network/datasets/pile.var/pile_uncopyrighted/pile-uncopyrighted \
  stage_pipeline.params.data.dataset_revision=null \
  stage_pipeline.params.data.text_field=text \
  stage_pipeline.params.data.tokenizer_name=EleutherAI/pythia-70m \
  stage_pipeline.params.data.train_split=train \
  stage_pipeline.params.data.val_split=validation \
  stage_pipeline.params.data.probe_split=validation \
  stage_pipeline.params.data.max_length=1024 \
  stage_pipeline.params.data.token_budget=5000000000 \
  stage_pipeline.params.probe.sampling=random \
  stage_pipeline.params.probe.seed=42 \
  stage_pipeline.params.probe.ids_path=null \
  stage_pipeline.params.probe.size=1024 \
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
  stage_pipeline.params.phate.random_state=42 \
  stage_pipeline.params.phate.n_pca=null \
  stage_pipeline.params.phate.n_landmark=null \
  stage_pipeline.params.phate.n_jobs=-1 \
  stage_pipeline.params.phate.verbose=false \
  stage_pipeline.params.phate.fit_fraction=1.0 \
  stage_pipeline.params.training.precision=bf16 \
  stage_pipeline.params.training.global_batch_size=512 \
  stage_pipeline.params.training.micro_batch_size=8 \
  stage_pipeline.params.training.grad_accum_steps=64 \
  stage_pipeline.params.training.max_steps=null \
  stage_pipeline.params.training.eval_every_n_steps=1000 \
  stage_pipeline.params.training.save_every_n_steps=1000 \
  stage_pipeline.params.training.save_top_k=3 \
  stage_pipeline.params.training.gradient_clip_norm=1.0 \
  stage_pipeline.params.optimizer.name=adamw \
  stage_pipeline.params.optimizer.learning_rate=3.0e-4 \
  'stage_pipeline.params.optimizer.betas=[0.9,0.95]' \
  stage_pipeline.params.optimizer.eps=1.0e-8 \
  stage_pipeline.params.optimizer.weight_decay=0.1 \
  stage_pipeline.params.lr_scheduler.type=cosine \
  stage_pipeline.params.lr_scheduler.warmup_steps=2000 \
  stage_pipeline.params.lr_scheduler.min_lr=3.0e-5 \
  stage_pipeline.params.regularization.dropout=0.0 \
  stage_pipeline.params.regularization.label_smoothing=0.0 \
  stage_pipeline.params.alignment.loss=mse \
  stage_pipeline.params.alignment.mse_reduction=mean \
  stage_pipeline.params.alignment.lambda_align=0.0 \
  stage_pipeline.params.alignment.lambda_schedule=constant \
  stage_pipeline.params.alignment.lambda_ramp_steps=0 \
  stage_pipeline.params.alignment.batch_size=16 \
  stage_pipeline.params.alignment.sample_every_n_steps=1 \
  stage_pipeline.params.seeds.global_seed=42 \
  stage_pipeline.params.seeds.model_init_seed=null \
  stage_pipeline.params.seeds.data_order_seed=null \
  stage_pipeline.params.seeds.dataloader_seed=null \
  stage_pipeline.params.seeds.phate_seed=42 \
  'stage_pipeline.params.sweep.teacher_model=[EleutherAI/pythia-1.4b]' \
  'stage_pipeline.params.sweep.student_model=[EleutherAI/pythia-70m]' \
  'stage_pipeline.params.sweep.seed=[42]' \
  'stage_pipeline.params.sweep.lambda_align=[0.0,0.5,1.0]' \
  'stage_pipeline.params.sweep.learning_rate=[3.0e-4]' \
  'stage_pipeline.params.sweep.max_length=[1024]' \
  'stage_pipeline.params.sweep.token_budget=[5000000000]'
