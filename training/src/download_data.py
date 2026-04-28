import kagglehub
import os
import shutil
from pathlib import Path
import logging

log = logging.getLogger(__name__)

def setup_data(copy_to_current_dir=False) -> Path:
    """Function for ensuring the data folder is available. If missing, download from kaggle.
        Args:
            copy_to_current_dir: bool - if data is not available, after downloading, ensure the
            folder with the images is copied to the current working directory (instead of .cache)
        
        Returns: Path to the dataset root directory
    """

    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', ''): # is on kaggle
        return Path('/kaggle/input/datasets/ikarus777/best-artworks-of-all-time')
    
    image_path = Path("training")/"data"/"images"
    if image_path.is_dir():
        log.info(f"{image_path} path exists")
        return image_path.parent
    
    path = kagglehub.dataset_download("ikarus777/best-artworks-of-all-time")
    log.info(f"Dataset files successfully downloaded to: {path}")

    if not copy_to_current_dir:
        return Path(path)
    shutil.copytree(Path(path)/"images", image_path)
    log.info(f"Dataset copied to: {image_path}")
    return image_path.parent