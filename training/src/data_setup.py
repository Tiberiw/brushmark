from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data import WeightedRandomSampler
from collections import Counter
from torch.utils.data import DataLoader
from painters_dataset import PaintersDataset
from interfaces import DataConfig
import hydra
import math
import logging

log = logging.getLogger(__name__)

def create_dataloaders(path: Path, pin_memory: bool, cfg: DataConfig):
    """Create train and validation dataloaders
        Args: 
            path: str - path to the data folder
        
        Returns:
            A tuple containing the train and validation dataloaders
    """
    train_transforms = hydra.utils.instantiate(cfg.train_transforms)
    val_transforms = hydra.utils.instantiate(cfg.val_transforms)
    train = PaintersDataset(path, train_transforms)
    valid = PaintersDataset(path, val_transforms)

    temp_train_idx, valid_idx = train_test_split(
        list(range(len(train))),
        stratify=train.labels, # stratified splitting
        test_size=cfg.valid_size,
        random_state=cfg.split_seed,
    )

    if not math.isclose(cfg.train_perc, 1.0):
        train_idx, _ = train_test_split(
            temp_train_idx,
            stratify=[train.labels[idx] for idx in temp_train_idx],
            train_size=cfg.train_perc,
            random_state=cfg.split_seed,
        )
    else:
        train_idx = temp_train_idx

    train_dataset = Subset(train, train_idx)
    valid_dataset = Subset(valid, valid_idx)
    log.info(f"Number of train images: {len(train_dataset)}")
    log.info(f"Number of validation images: {len(valid_dataset)}")

    targets = [train.labels[idx] for idx in train_idx]
    class_counts = Counter(targets)
    total = len(targets)
    # Linear inverse: aggressive oversampling so minority classes appear ~ equally often
    class_weights = {cls: total / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[target] for target in targets]
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataloader.batch_size,
        sampler=train_sampler,
        pin_memory=pin_memory,
        num_workers=cfg.dataloader.num_workers
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=cfg.dataloader.num_workers
    )

    return train_loader, valid_loader, train.idx_to_class, class_weights