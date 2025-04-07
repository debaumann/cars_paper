#!/bin/bash
#SBATCH --job-name=my_python_job  # Job name
#SBATCH --account=ls_polle        # SLURM account
#SBATCH --output=output_attention_free_100.log        # Standard output and error log
#SBATCH --error=error_attention_free_100.log          # Standard error log
#SBATCH --time=08:00:00            # Time limit (hh:mm:ss)
#SBATCH --ntasks=1                 # Number of tasks (4 processes)
#SBATCH --cpus-per-task=4          # Number of CPU cores per task
#SBATCH --mem-per-cpu=4G           # Memory per CPU (total = 4*4G = 16G)
#SBATCH --gpus=rtx_4090:1          # Request 1 RTX 4090 GPU
#SBATCH --gres=gpumem:14g          # Request 14GB GPU memory
#SBATCH --partition=standard        # Adjust if needed

# Load environment variables (ensures conda works)
source ~/.bashrc  

# Activate Conda environment
conda activate transformer  # Replace with your Conda environment name

# Run the Python script
python attention_vit_free.py --weight 100 # Replace with your actual script

