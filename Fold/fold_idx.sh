#!/bin/bash

#SBATCH --job-name=cv_job_atlas_64
#SBATCH --account=llonpp
#SBATCH --output=logs/fold_%A_%a.out
#SBATCH --error=logs/error_fold_%A_%a.err
#SBATCH --ntasks-per-node=72
#SBATCH --nodes=1
#SBATCH --time 01:00:00
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --mem=40000M
#SBATCH --export=ALL
#SBATCH --array=0-4

cd /data/leuven/369/vsc36935/SLURM/Fold
source /data/leuven/369/vsc36935/miniconda3/etc/profile.d/conda.sh
conda activate neo_demo
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_fold.py $SLURM_ARRAY_TASK_ID
