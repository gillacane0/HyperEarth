import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, recall_score


def compute_metrics(y_true, y_pred) -> dict[str, float | np.ndarray]:
    """Standard hyperspectral classification metrics.

    OA = overall accuracy, AA = average of the per-class accuracies, Kappa = Cohen's kappa.
    """
    return {
        "OA": accuracy_score(y_true, y_pred),
        "AA": balanced_accuracy_score(y_true, y_pred),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "per_class": recall_score(y_true, y_pred, average=None),
    }

def print_metrics(metrics: dict[str, float | np.ndarray]) -> None:
    """Print the metrics in a human-readable format."""
    print(f"  OA:    {metrics['OA']:.2%}")
    print(f"  AA:    {metrics['AA']:.2%}")
    print(f"  Kappa: {metrics['Kappa']:.4f}")
    print("  Per-class accuracy:")
    for cls, acc in enumerate(metrics["per_class"], start=1):
        print(f"    class {cls:2d}: {acc:.2%}")
    