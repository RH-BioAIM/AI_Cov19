#!/bin/sh
#SBATCH --job-name=sd-EmbedExtract
#SBATCH --output=/path/to/logs/extract_embeddings-%j.out
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00

module load Anaconda3
conda activate hunterAI

cd /path/to/repo/fusion_experiments
python -u extract_embeddings.py
