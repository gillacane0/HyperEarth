import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from hyperearth.training.metrics import compute_metrics


def get_device():
    """Use the GPU if PyTorch can see one, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, device=None) -> tuple[nn.Module, list[dict]]:
    """Train a classification model and keep the weights of the epoch with the best validation accuracy."""
    device = device or get_device()
    model.to(device)
    print(f"Training on {device} for {epochs} epochs...")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_state = -1.0, None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y)
        train_loss = running_loss / len(train_loader.dataset)

        y_true, y_pred = predict(model, val_loader)
        val_acc = accuracy_score(y_true, y_pred)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})

        # Keep a copy of the best weights seen so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch:3d}/{epochs} | train loss {train_loss:.4f} | val acc {val_acc:.2%}")

    print(f"Best validation accuracy: {best_val_acc:.2%}")
    model.load_state_dict(best_state)
    return model, history

@torch.no_grad()
def predict(model, loader) -> tuple[np.ndarray, np.ndarray]:
    """Return the true and the predicted classes of every sample in the loader."""
    device = next(model.parameters()).device
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        y_pred.append(model(x.to(device)).argmax(dim=1).cpu())
        y_true.append(y)
    return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()


def run_experiment(model, dataset, epochs=50, lr=1e-3) -> tuple[nn.Module, list[dict], dict]:
    """Run a complete experiment: train on the dataset, then evaluate on its test set.

    Returns (model, history, metrics).
    """

    train_loader, val_loader, test_loader = dataset.get_loaders()
    model, history = train_model(model, train_loader, val_loader, epochs=epochs, lr=lr)
    y_true, y_pred = predict(model, test_loader)
    metrics = compute_metrics(y_true, y_pred)
    return model, history, metrics
