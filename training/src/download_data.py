# Ensure we have the data available
import kagglehub
import os
import shutil
from pathlib import Path

def setup_data(copy_to_current_dir=False) -> Path:
    """Function to ensure the dataset is available. If not, download from kaggle"""

    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', ''): # is on kaggle
        return Path('/kaggle/input/datasets/ikarus777/best-artworks-of-all-time')
    
    image_path = Path("training")/"data"/"images"
    if image_path.is_dir():
        print(f"{image_path} path exists")
        return image_path.parent
    
    path = kagglehub.dataset_download("ikarus777/best-artworks-of-all-time")
    print(f"Dataset files successfully downloaded to: {path}")

    if not copy_to_current_dir:
        return Path(path)
    shutil.copytree(f'{path}/images', image_path)
    print(f"Dataset copied to: {image_path}")
    return image_path.parent