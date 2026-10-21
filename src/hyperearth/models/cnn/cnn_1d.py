import torch.nn as nn


class CNN1D(nn.Module):
    """1D CNN for pixel-wise hyperspectral classification.

    Each pixel is classified from its spectrum alone: a sequence of `num_bands` values in wavelength
    order, along which the convolution filters slide. The bands must keep their order, so the data
    must not go through the PCA (n_components=None).

    Input: (batch, bands, patch_size, patch_size); only the center pixel is used, so patch_size=1
    is the natural choice.
    Output: (batch, num_classes) scores for the class of the pixel.
    """

    def __init__(self, num_bands, num_classes, dropout=0.4):
        super().__init__()

        # Feature extractor: each filter looks at a few neighbouring bands (7, then 5, then 3)
        # and learns local shapes of the spectrum, such as peaks, dips and steep slopes
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # The two MaxPool1d(2) halve the length of the spectrum twice: e.g. 220 bands -> 110 -> 55
        reduced_length = num_bands // 4

        # Classifier: Flatten keeps the position of every feature along the spectrum, because
        # (unlike in images) where a pattern appears, i.e. at which wavelength, matters
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * reduced_length, num_classes),
        )

    def forward(self, x):
        # Spectrum of the center pixel as a 1-channel sequence: (batch, bands, h, w) -> (batch, 1, bands)
        x = x[:, :, x.shape[2] // 2, x.shape[3] // 2].unsqueeze(1)
        return self.classifier(self.features(x))
