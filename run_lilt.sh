#!/bin/bash

#SBATCH --job-name=lilt
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=8
#SBATCH --output=lilt_out/lilt_bitfit_LWA1.out

source /home/yunjinna/.bashrc
#source activate miss

srun python run_LilT.py with task_lilt_mmimdb exp_name=finetune_lilt_bitfit_LWA1 bitfit=True