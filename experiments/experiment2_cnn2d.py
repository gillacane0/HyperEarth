import torch

from hyperearth.data.dataloader.hyperdataset import HyperDataset
from hyperearth.models.cnn.cnn_2d import CNN2D
from hyperearth.training.trainer import train_model, predict
from hyperearth.training.metrics import compute_metrics


# Experiment settings
N_COMPONENTS = 30
PATCH_SIZE = 11     # side of the square patch around each pixel (must be odd)
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
    train_loader, val_loader, test_loader = dataset.get_loaders()

    model = CNN2D(in_channels=dataset.num_bands, num_classes=dataset.num_classes)
    model, _ = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR)

    y_true, y_pred = predict(model, test_loader)
    metrics = compute_metrics(y_true, y_pred)

    print("\nTest results:")
    print(f"  OA:    {metrics['OA']:.2%}")
    print(f"  AA:    {metrics['AA']:.2%}")
    print(f"  Kappa: {metrics['Kappa']:.4f}")
    print("  Per-class accuracy:")
    for cls, acc in enumerate(metrics["per_class"], start=1):
        print(f"    class {cls:2d}: {acc:.2%}")


if __name__ == "__main__":
    main()
