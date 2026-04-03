#!/bin/bash
# Tiny end-to-end smoke test for config-native distillation pipeline.
#
# Submit:
#   sbatch downstream/distill_family_study/scripts/run_pythia_distill_tiny_smoke.sbatch

#SBATCH --job-name=distill_smoke
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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

python -m manylatents.main \
  "hydra.searchpath=[file://${DOWNSTREAM_CONFIG_DIR}]" \
  experiment=pythia_distill_sweep_template \
  name=pythia_distill_tiny_smoke_${SLURM_JOB_ID} \
  output_dir=./outputs \
  stage_pipeline.resume=false \
  stage_pipeline.params.data.dataset_name=wikitext \
  stage_pipeline.params.data.dataset_config=wikitext-2-raw-v1 \
  stage_pipeline.params.data.dataset_path=null \
  stage_pipeline.params.data.dataset_revision=null \
  stage_pipeline.params.data.text_field=text \
  stage_pipeline.params.data.train_split=train \
  stage_pipeline.params.data.val_split=validation \
  stage_pipeline.params.data.probe_split=validation \
  stage_pipeline.params.data.tokenizer_name=sshleifer/tiny-gpt2 \
  stage_pipeline.params.data.max_length=32 \
  stage_pipeline.params.probe.size=8 \
  stage_pipeline.params.teacher.model_name_or_path=sshleifer/tiny-gpt2 \
  stage_pipeline.params.teacher.model_revision=main \
  stage_pipeline.params.student.model_name_or_path=sshleifer/tiny-gpt2 \
  stage_pipeline.params.student.model_revision=main \
  stage_pipeline.params.student.penultimate_dim=2 \
  stage_pipeline.params.phate.n_components=2 \
  stage_pipeline.params.training.global_batch_size=2 \
  stage_pipeline.params.training.micro_batch_size=1 \
  stage_pipeline.params.training.grad_accum_steps=2 \
  stage_pipeline.params.training.max_steps=1 \
  stage_pipeline.params.training.eval_every_n_steps=1 \
  stage_pipeline.params.training.save_every_n_steps=1 \
  stage_pipeline.params.training.save_top_k=1 \
  'stage_pipeline.params.sweep.teacher_model=[sshleifer/tiny-gpt2]' \
  'stage_pipeline.params.sweep.student_model=[sshleifer/tiny-gpt2]' \
  'stage_pipeline.params.sweep.seed=[42]' \
  'stage_pipeline.params.sweep.lambda_align=[0.0,0.3]' \
  'stage_pipeline.params.sweep.learning_rate=[1e-4]' \
  'stage_pipeline.params.sweep.max_length=[32]' \
  'stage_pipeline.params.sweep.token_budget=[64]'
