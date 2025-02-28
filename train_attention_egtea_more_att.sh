#!/bin/bash
#SBATCH --account=ls_polle
#SBATCH --ntasks=4
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=rtx_4090:1
#SBATCH --gres=gpumem:10g
#SBATCH --job-name=attention_vit_job
#SBATCH --output=attention_vit_%j.out

# Load necessary modules or activate your conda environment
 # or your preferred module system
source activate transformer  # adjust to your conda environment name


# Change directory to where your script is located

cd /cluster/home/debaumann/cars_paper/training

# Run your attention_vit.py script
python attention_vit_egtea_more_att.py
