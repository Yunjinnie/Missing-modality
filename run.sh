#!/bin/bash

#SBATCH --job-name=prompt0
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=8
#SBATCH --output=out/prompt0_.out

source /home/yunjinna/.bashrc
#source activate miss

srun python run.py with data_root=./datasets/mmimdb learnt_p=False full_finetune=False prompt_use=False exp_name=prompt0_finetune_text task_finetune_mmimdb
