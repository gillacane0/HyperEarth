import torch.nn as nn


class CNN2D(nn.Module):
    """Simple 2D CNN for patch-based hyperspectral classification.

    Input: (batch, bands, patch_size, patch_size), e.g. the PCA components of each patch.
    Output: (batch, num_classes) scores for the class of the center pixel.
    """

    def __init__(self, in_channels, num_classes, dropout=0.4):
        super().__init__()

        # Feature extractor: 3x3 convolutions mix neighbouring pixels and spectral channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # one value per channel, whatever the patch size
        )

        # Classifier: from the 256 features to one score per class
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
