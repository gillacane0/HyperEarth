from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, recall_score


def compute_metrics(y_true, y_pred):
    """Standard hyperspectral classification metrics.

    OA = overall accuracy, AA = average of the per-class accuracies, Kappa = Cohen's kappa.
    """
    return {
        "OA": accuracy_score(y_true, y_pred),
        "AA": balanced_accuracy_score(y_true, y_pred),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "per_class": recall_score(y_true, y_pred, average=None),
    }
