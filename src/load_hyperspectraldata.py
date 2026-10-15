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
    # n_components can be an int (number of components) or a float in (0, 1) (fraction of variance to keep)
    pca = PCA(n_components=n_components)
    data_reduced = pca.fit_transform(data_normalized)
    print(f"PCA: {pca.n_components_} components retain "
          f"{pca.explained_variance_ratio_.sum():.2%} of the variance")

    # Reshape back to the original spatial dimensions with reduced spectral bands
    data_preprocessed = data_reduced.reshape(original_shape[0], original_shape[1], pca.n_components_)

    return data_preprocessed, pca

def visualize_data(data, data_preprocessed, labels, band=43, component=0):
    """Data visualization: a raw spectral band, a PCA component and the label map.

    The default band 43 is ~803 nm (near-infrared), where vegetation types differ the most.
    """
    print("Data visualization...")

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(data[:, :, band], cmap='viridis')
    plt.title(f"Raw spectral band {band}")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(data_preprocessed[:, :, component], cmap='viridis')
    plt.title(f"PCA component {component}")
    plt.colorbar()

    # Hide label 0 (unlabeled pixels) and use one distinct color per class (1-16)
    plt.subplot(1, 3, 3)
    plt.imshow(np.ma.masked_equal(labels, 0), cmap=plt.get_cmap('tab20', 16), vmin=0.5, vmax=16.5)
    plt.title("Label map (white = unlabeled)")
    plt.colorbar(ticks=range(1, 17))

    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/sample_band.png")
    plt.close()

    print("Data visualization saved in results/plots/sample_band.png")

def main():
    
    data, labels = load_indian_pines()
    data_preprocessed, pca = preprocess_data(data)
    visualize_data(data, data_preprocessed, labels)


if __name__ == "__main__":
    main()
