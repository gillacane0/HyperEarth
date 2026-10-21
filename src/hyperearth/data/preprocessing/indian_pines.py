from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def preprocess_data(data, n_components=30):
    """Preprocess the data: normalization and PCA dimensionality reduction.

    With n_components=None the PCA is skipped and all the original bands are kept (only normalized),
    e.g. for 1D CNNs, which need the bands in wavelength order.
    """
    print("Preprocessing...")

    # Reshape the data to 2D for PCA
    original_shape = data.shape
    data_2d = data.reshape(-1, original_shape[-1])

    # Normalization of the data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data_2d)

    if n_components is None:
        print(f"PCA skipped: keeping all the {original_shape[-1]} bands")
        return data_normalized.reshape(original_shape), None

    # Apply PCA for dimensionality reduction
    # n_components can be an int (number of components) or a float in (0, 1) (fraction of variance to keep)
    pca = PCA(n_components=n_components)
    data_reduced = pca.fit_transform(data_normalized)
    print(f"PCA: {pca.n_components_} components retain "
          f"{pca.explained_variance_ratio_.sum():.2%} of the variance")

    # Reshape back to the original spatial dimensions with reduced spectral bands
    data_preprocessed = data_reduced.reshape(original_shape[0], original_shape[1], pca.n_components_)

    return data_preprocessed, pca
