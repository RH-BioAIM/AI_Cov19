import kagglehub
import shutil
import os

# Download dataset to default kagglehub cache
path = kagglehub.dataset_download("tawsifurrahman/covid19-radiography-database")

# Your desired path
target_dir = "/path/to/data/pos_neg_database"

# Create target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Copy dataset to your target directory
shutil.copytree(path, target_dir, dirs_exist_ok=True)

print("Dataset saved to:", target_dir)
