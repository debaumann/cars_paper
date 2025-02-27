#!/bin/bash
#SBATCH --account=ls_polle
#SBATCH --ntasks=4
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=egtea_job
#SBATCH --output=egtea_%j.out

# Load necessary modules or activate your conda environment
 # or your preferred module system
source activate transformer  # adjust to your conda environment name


# Change directory to where your script is located
cd /cluster/home/debaumann/cars_paper/dataset

# Run your attention_vit.py script
python egtea.py
