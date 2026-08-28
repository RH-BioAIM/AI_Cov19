#!/bin/sh
#SBATCH --job-name=sd-CRTraining
#SBATCH --output=/path/to/logs/run-job-%j.txt
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


conda activate hunterAI
python train_swin.py

