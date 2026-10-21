import torch

from hyperearth.data.dataloader.hyperdataset import HyperDataset
from hyperearth.models.cnn.cnn_1d import CNN1D
from hyperearth.training.metrics import print_metrics
from hyperearth.training.trainer import run_experiment


# Experiment settings
N_COMPONENTS = None  # no PCA: the 1D CNN needs the original bands in wavelength order
PATCH_SIZE = 1       # single pixels: the 1D CNN only looks at the spectrum
TRAIN_RATIO = 0.2
VAL_RATIO = 0.1
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
SEED = 42


def main():
    torch.manual_seed(SEED)

    dataset = HyperDataset("indian_pines", n_components=N_COMPONENTS, patch_size=PATCH_SIZE,
                           train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, batch_size=BATCH_SIZE, seed=SEED)

    model = CNN1D(num_bands=dataset.num_bands, num_classes=dataset.num_classes)

    model, _, metrics = run_experiment(model, dataset, epochs=EPOCHS, lr=LR)

    print_metrics(metrics)


if __name__ == "__main__":
    main()
