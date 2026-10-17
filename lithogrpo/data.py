from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def image_tensor(path, size):
    with Image.open(path) as image:
        image = image.convert('L').resize((size, size), Image.Resampling.NEAREST)
        return torch.from_numpy(np.array(image, dtype=np.float32)/255)[None]


class LayoutPairs(Dataset):
    """LithoBench target/*.png + pixelILT/*.png, sorted deterministic 90/10 split."""
    def __init__(self, root, size=512, split='train', ratio=.9, paired=True):
        self.root, self.size, self.paired = Path(root), size, paired
        self.files = sorted((self.root/'target').glob('*.png'))
        if not self.files:
            raise FileNotFoundError(f'No target/*.png in {root}')
        if paired:
            missing = [p.name for p in self.files if not (self.root/'pixelILT'/p.name).exists()]
            if missing:
                raise FileNotFoundError(f'Missing {len(missing)} paired masks, e.g. {missing[0]}')
        cut = round(len(self.files)*ratio)
        if split == 'train':
            self.files = self.files[:cut]
        elif split == 'test':
            self.files = self.files[cut:]
        elif split != 'all':
            raise ValueError('split must be train, test or all')
        if not self.files:
            raise ValueError(f'Empty {split} split; supply more data or --split all')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        target = image_tensor(path, self.size)
        mask = image_tensor(self.root/'pixelILT'/path.name, self.size) if self.paired else target
        return target, mask, path.stem


def synthetic_data(root, count=20, size=64, seed=1):
    """Synthetic rectangles only, with target-as-mask labels, NOT benchmark training data."""
    root = Path(root)
    for name in ('target', 'pixelILT'):
        (root/name).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(count):
        a = np.zeros((size, size), np.uint8)
        for _ in range(int(rng.integers(1, 5))):
            y, x = rng.integers(size//8, size//2, size=2)
            h, w = rng.integers(max(2, size//8), max(3, size//3), size=2)
            a[y:y+h, x:x+w] = 255
        for name in ('target', 'pixelILT'):
            path = root/name/f'{i:05d}.png'
            if path.exists():
                raise FileExistsError(path)
            Image.fromarray(a).save(path)
