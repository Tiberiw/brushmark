from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import unicodedata
import torch

from torchvision import transforms

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from torch.utils.data import WeightedRandomSampler
from collections import Counter

from torch.utils.data import DataLoader

class PaintersDataset(Dataset):
    """ PyTorch Dataset for the 'Best Artworks of All Time' Kaggle dataset
        Args:
        root: Path to the Kaggle dataset root. Images are loaded from
            root/images/images/<Artist_Name>/*.jpg
    """

    def __init__(self, root: Path | str, transform=None):
        root = Path(root)
        self.transform = transform
        data_dir = root / 'images' / 'images'

        if not data_dir.exists():
            raise FileNotFoundError(
                f"Could not find {data_dir}. "
                f"Pass the dataset root — images are expected at root/images/images/<Artist_Name>/*.jpg"
            )
        self.class_names = []
        self.image_paths = []
        self.labels = []
        
        # Mapping for folders whose names got mangled (encoding issues).
        LABEL_ALIASES = {
            "Albrecht_DuΓòá├¬rer": "Albrecht_Dürer",
            "Albrecht_Du╠êrer": "Albrecht_Dürer"
        }
        for folder in sorted(data_dir.iterdir()):
            if not folder.is_dir():
                continue
            label = self._normalize(LABEL_ALIASES.get(folder.name, folder.name))
            if label in self.class_names:
                continue # Skip duplicate folders (e.g. for "Albrecht_Dürer")
            self.class_names.append(label)
            for img_path in folder.glob("*.jpg"):
                self.image_paths.append(img_path)
                self.labels.append(label)
                
        self.class_to_idx = {cls:idx for idx, cls in enumerate(self.class_names)}
        self.labels = [self.class_to_idx[label] for label in self.labels]
        self.idx_to_class = {idx:cls for cls, idx in self.class_to_idx.items()}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
          img = self.transform(img)
        return img, self.labels[idx]

    def _normalize(self, s: str) -> str:
        return unicodedata.normalize('NFC', s)

def create_dataloaders(path: Path, pin_memory: bool, bs: int = 32, num_workers: int = 4):
    """Create train and validation dataloaders
        Args: 
            path: str - path to the data folder
        
        Returns:
            A tuple containing the train and validation dataloaders
    """

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(448, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.3)
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(540),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )    
    ])

    train = PaintersDataset(path, train_transforms)
    valid = PaintersDataset(path, val_transforms)

    # We split the data into train (85%) and validation (15%) using stratified sampling.
    indices = list(range(len(train)))
    train_idx, valid_idx = train_test_split(
        indices,
        stratify=train.labels,
        test_size=0.15,
        random_state=42
    )
    train_dataset = Subset(train, train_idx)
    valid_dataset = Subset(valid, valid_idx)
    print(f"Number of train images: {len(train_dataset)}")
    print(f"Number of validation images: {len(valid_dataset)}")

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
        batch_size=bs,
        sampler=train_sampler,
        pin_memory=pin_memory,
        num_workers=num_workers
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=bs,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers
    )

    return train_loader, valid_loader, train.idx_to_class, class_weights