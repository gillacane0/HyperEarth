import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def split_labeled_pixels(labels, train_ratio=0.2, val_ratio=0.1, seed=42):
    """Split the labeled pixels (label > 0) into train/val/test sets, stratified by class.

    Returns three arrays of (row, col) coordinates.
    """
    coords = np.argwhere(labels > 0)
    classes = labels[coords[:, 0], coords[:, 1]]

    train_coords, rest_coords, _, rest_classes = train_test_split(
        coords, classes, train_size=train_ratio, stratify=classes, random_state=seed)
    val_coords, test_coords = train_test_split(
        rest_coords, train_size=val_ratio / (1 - train_ratio), stratify=rest_classes, random_state=seed)

    return train_coords, val_coords, test_coords


class PatchDataset(Dataset):
    """Square patches centered on the given pixels, labeled with the class of the center pixel.

    Each item is a (bands, patch_size, patch_size) tensor and a class index in 0..num_classes-1
    (the original labels start at 1, because 0 means unlabeled).
    """

    def __init__(self, cube, labels, coords, patch_size=11):
        if patch_size % 2 == 0:
            raise ValueError("patch_size must be odd, so that the patch has a center pixel")
        self.patch_size = patch_size
        margin = patch_size // 2

        # Pad the borders so that the patches around edge pixels are complete
        padded = np.pad(cube, ((margin, margin), (margin, margin), (0, 0)), mode="reflect")
        self.cube = torch.from_numpy(padded).permute(2, 0, 1).contiguous().float()  # (bands, rows, cols)

        self.coords = coords
        self.targets = torch.from_numpy(labels[coords[:, 0], coords[:, 1]].astype(np.int64) - 1)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        # Because of the padding, pixel (row, col) is the center of cube[:, row:row+p, col:col+p]
        row, col = self.coords[idx]
        patch = self.cube[:, row:row + self.patch_size, col:col + self.patch_size]
        return patch, self.targets[idx]
