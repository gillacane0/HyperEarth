import numpy as np
import kagglehub

from .utils import read_tif, find_file


KAGGLE_DATASET = "ankitgaidhar/indian-pines-hyperspectral-dataset"


def load_indian_pines():
    """Load the Indian Pines dataset from Kaggle via kagglehub.

    The dataset is downloaded on first use and cached in ~/.cache/kagglehub,
    so later calls reuse the local copy.
    """
    print(f"Downloading/locating dataset {KAGGLE_DATASET}...")
    dataset_dir = kagglehub.dataset_download(KAGGLE_DATASET)

    data_path = find_file(dataset_dir, "19920612_AVIRIS_IndianPine_Site3.tif")
    labels_path = find_file(dataset_dir, "19920612_AVIRIS_IndianPine_Site3_gr.tif")

    print("Loading data and labels...")
    data = read_tif(data_path).transpose(1, 2, 0).astype(np.float32)  # (145, 145, 220)
    labels = read_tif(labels_path)[0]  # (145, 145)

    return data, labels
