# Ensure we have the data available
import kagglehub
import os
import shutil

path = '/kaggle/input/datasets/ikarus777/best-artworks-of-all-time'
if not os.environ.get('KAGGLE_KERNEL_RUN_TYPE', ''):
    path = kagglehub.dataset_download("ikarus777/best-artworks-of-all-time")
print(f"Path to dataset files: {path}")
shutil.copytree(f'{path}/images', "data/images")
print(f"Dataset copied to: ./data")