import os
import numpy as np
import matplotlib.pyplot as plt

from hyperearth.data.dataloader.loaders import load_indian_pines
from hyperearth.data.preprocessing.indian_pines import preprocess_data


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
