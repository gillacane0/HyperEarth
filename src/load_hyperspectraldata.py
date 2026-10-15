import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import glob
import warnings
import kagglehub
import rasterio
from rasterio.errors import NotGeoreferencedWarning

KAGGLE_DATASET = "ankitgaidhar/indian-pines-hyperspectral-dataset"

def _find_file(root, filename):
    """Find a file by name anywhere inside the downloaded dataset folder."""
    matches = glob.glob(os.path.join(root, "**", filename), recursive=True)
    if not matches:
        raise FileNotFoundError(f"{filename} not found in {root}")
    return matches[0]

def _read_tif(path):
    """Read a (non-georeferenced) TIFF as a (bands, rows, cols) array."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as src:
            return src.read()

def load_indian_pines():
    """Load the Indian Pines dataset from Kaggle via kagglehub.

    The dataset is downloaded on first use and cached in ~/.cache/kagglehub,
    so later calls reuse the local copy.
    """
    print(f"Downloading/locating dataset {KAGGLE_DATASET}...")
    dataset_dir = kagglehub.dataset_download(KAGGLE_DATASET)

    # Indian Pines scene (145x145 pixels, 220 bands) and its ground reference (16 classes, 0 = unlabeled)
    data_path = _find_file(dataset_dir, "19920612_AVIRIS_IndianPine_Site3.tif")
    labels_path = _find_file(dataset_dir, "19920612_AVIRIS_IndianPine_Site3_gr.tif")

    print("Loading data and labels...")
    data = _read_tif(data_path).transpose(1, 2, 0).astype(np.float32)  # (145, 145, 220)
    labels = _read_tif(labels_path)[0]  # (145, 145)

    return data, labels

def preprocess_data(data, n_components=30):
    """Preprocess the data: normalization and PCA dimensionality reduction."""
    print("Preprocessing...")

    # Reshape the data to 2D for PCA
    original_shape = data.shape
    data_2d = data.reshape(-1, original_shape[-1])

    # Normalization of the data 
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data_2d)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=n_components)
    data_reduced = pca.fit_transform(data_normalized)

    # Reshape back to the original spatial dimensions with reduced spectral bands
    data_preprocessed = data_reduced.reshape(original_shape[0], original_shape[1], n_components)

    return data_preprocessed, pca

def visualize_data(data, labels, sample_band=10):
    """Data visualization: display a sample spectral band and the label map."""
    print("Data visualization...")

    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(data[:, :, sample_band], cmap='viridis')
    plt.title(f"Spectral band {sample_band}")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(labels, cmap='jet', alpha=0.7)
    plt.title("Label map")
    plt.colorbar()

    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/sample_band.png")
    plt.close()

    print("Data visualization saved in results/plots/sample_band.png")

def main():
    
    data, labels = load_indian_pines()
    data_preprocessed, pca = preprocess_data(data)
    visualize_data(data_preprocessed, labels)


if __name__ == "__main__":
    main()
