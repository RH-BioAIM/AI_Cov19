#!/bin/sh
#SBATCH --job-name=sd-CRTraining
#SBATCH --output=/path/to/logs/pos_neg_analysis_job-%j.txt
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2                
#SBATCH --nodes=1  
#SBATCH --ntasks=16                 # Use 16 CPU cores
#SBATCH --mem-per-cpu=8G            # More memory per CPU
#SBATCH --time=12:00:00             # 12 hours max
#SBATCH --mail-type=BEGIN,END 
#SBATCH --mail-user=your-email@example.com  

#SBATCH --mail-user=your-email@example.com
#SBATCH --mail-type=BEGIN,END

module load Anaconda3
conda activate hunterAI
python pos_neg_analysis.py

