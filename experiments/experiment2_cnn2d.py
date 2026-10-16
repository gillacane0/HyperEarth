import torch
from torch.utils.data import DataLoader

from hyperearth.data.loaders import load_indian_pines
from hyperearth.data.preprocessing.indian_pines import preprocess_data
from hyperearth.data.patches import PatchDataset, split_labeled_pixels
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

    data, labels = load_indian_pines()
    data_pca, _ = preprocess_data(data, n_components=N_COMPONENTS)

    train_coords, val_coords, test_coords = split_labeled_pixels(labels, TRAIN_RATIO, VAL_RATIO, seed=SEED)
    print(f"Pixels: {len(train_coords)} train, {len(val_coords)} val, {len(test_coords)} test")

    train_loader = DataLoader(PatchDataset(data_pca, labels, train_coords, PATCH_SIZE),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(PatchDataset(data_pca, labels, val_coords, PATCH_SIZE), batch_size=BATCH_SIZE)
    test_loader = DataLoader(PatchDataset(data_pca, labels, test_coords, PATCH_SIZE), batch_size=BATCH_SIZE)

    num_classes = int(labels.max())  # 16 (label 0 = unlabeled is excluded)
    model = CNN2D(in_channels=N_COMPONENTS, num_classes=num_classes)
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
