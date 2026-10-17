from torch.utils.data import DataLoader

from .loaders import load_indian_pines
from .patches import PatchDataset, split_labeled_pixels
from ..preprocessing.indian_pines import preprocess_data


DATASETS = {
    "indian_pines": (load_indian_pines, preprocess_data),
}


class HyperDataset:
    """Prepares a hyperspectral dataset by name and builds its train/val/test DataLoaders.

    It loads the raw data, preprocesses it, splits the labeled pixels and cuts the patches.
    After creation, `num_bands` and `num_classes` tell the model what input and output size to use.
    """

    def __init__(self, name, n_components=30, patch_size=11, train_ratio=0.2, val_ratio=0.1,
                 batch_size=64, seed=42):
        
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset '{name}'. Available: {', '.join(DATASETS)}")
        load_fn, preprocess_fn = DATASETS[name]

        self.name = name
        self.patch_size = patch_size
        self.batch_size = batch_size

        data, self.labels = load_fn()
        self.data, self.pca = preprocess_fn(data, n_components=n_components)

        self.num_bands = self.data.shape[-1]
        self.num_classes = int(self.labels.max())  # label 0 = unlabeled is not a class

        self.train_coords, self.val_coords, self.test_coords = split_labeled_pixels(
            self.labels, train_ratio, val_ratio, seed=seed)
        print(f"Pixels: {len(self.train_coords)} train, {len(self.val_coords)} val, "
              f"{len(self.test_coords)} test")

    def get_loaders(self):
        """Return the (train, val, test) DataLoaders. Only the train loader shuffles the samples."""
        return (
            self._make_loader(self.train_coords, shuffle=True),
            self._make_loader(self.val_coords),
            self._make_loader(self.test_coords),
        )

    def _make_loader(self, coords, shuffle=False):
        dataset = PatchDataset(self.data, self.labels, coords, self.patch_size)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
