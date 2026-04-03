#!/bin/bash
# Cheap final validation suite for the publication study.
#
# Runs:
# - targeted unit/config tests
# - study manifest dry run
# - verification of known passing family smoke logs
# - markdown/json report generation

#SBATCH --job-name=distill_final_validation
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --time=00:30:00
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

python downstream/distill_family_study/scripts/final_validation_report.py
