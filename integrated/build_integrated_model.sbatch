#!/bin/sh
#SBATCH --job-name=sd-IntegratedModel
#SBATCH --output=/path/to/logs/run-job-%j.out
#
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=your-email@example.com

module load Anaconda3
conda activate hunterAI

python build_integrated_model.py
