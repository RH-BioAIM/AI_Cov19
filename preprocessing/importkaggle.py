"""
Downloads the COVID-19 Radiography Database from Kaggle and copies it to a
local directory.

Input: the Kaggle dataset tawsifurrahman/covid19-radiography-database.
Output: a local copy of the dataset at the configured target directory.
"""
import kagglehub
import shutil
import os

# Download dataset to the default kagglehub cache.
path = kagglehub.dataset_download("tawsifurrahman/covid19-radiography-database")

# Target directory for the local copy.
target_dir = "/path/to/data/pos_neg_database"

# Create the target directory if it doesn't exist.
os.makedirs(target_dir, exist_ok=True)

# Copy the dataset to the target directory.
shutil.copytree(path, target_dir, dirs_exist_ok=True)

print("Dataset saved to:", target_dir)
