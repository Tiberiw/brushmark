from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import unicodedata

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