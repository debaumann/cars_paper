#!/bin/bash
#SBATCH --account=ls_polle
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=6G
#SBATCH --gpus=rtx_4090:1
#SBATCH --gres=gpumem:16g
#SBATCH --job-name=interleave
#SBATCH --output=interleave.out

# Load necessary modules or activate your conda environment
 # or your preferred module system
source activate transformer  # adjust to your conda environment name


# Change directory to where your script is located
cd /cluster/home/debaumann/cars_paper/training

# Run your attention_vit.py script
python attention_vit_interleave.py
