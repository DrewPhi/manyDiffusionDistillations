#!/bin/bash
# Sequential mini-launch validation for submit_distill_study.py.
#
# This job:
# - materializes the 6-run per-family mini launcher slice
# - submits each run sequentially with sbatch --wait
# - aggregates and plots the resulting real outputs
# - writes a final markdown/json report

#SBATCH --job-name=distill_mini_launch_validation
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

cd /home/mila/d/drewd/newManylatents/manylatents
mkdir -p outputs/slurm

module load miniconda
export PS1=""
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u
conda activate manylatents

export WANDB_ENTITY=${WANDB_ENTITY:-cesar-valdez-mcgill-university}
export WANDB_PROJECT=${WANDB_PROJECT:-diffusion-distillation}
export WANDB_MODE=${WANDB_MODE:-online}

python downstream/distill_family_study/scripts/run_submit_distill_study_mini_validation.py
