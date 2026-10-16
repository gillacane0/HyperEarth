import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from hyperearth.data.patches import PatchDataset, split_labeled_pixels
from hyperearth.models.cnn.cnn_2d import CNN2D
from hyperearth.training.metrics import compute_metrics
from hyperearth.training.trainer import predict, train_model


NUM_CLASSES = 3
BANDS = 8


@pytest.fixture
def synthetic_scene():
    """Small fake hyperspectral scene (30x30 pixels, 8 bands) that is easy to classify.

    3 classes in vertical stripes, each with its own spectrum plus a little noise.
    The first 3 rows are unlabeled (0), like the background in Indian Pines.
    """
    rng = np.random.default_rng(0)
    labels = np.zeros((30, 30), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        labels[:, c * 10:(c + 1) * 10] = c + 1
    labels[:3, :] = 0

    spectra = rng.normal(size=(NUM_CLASSES + 1, BANDS))
    cube = spectra[labels] + 0.1 * rng.normal(size=(30, 30, BANDS))
    return cube.astype(np.float32), labels


def _as_set(coords):
    return {tuple(rc) for rc in coords}


# --- Data split ---

def test_split_covers_all_labeled_pixels_without_overlap(synthetic_scene):
    _, labels = synthetic_scene
    train, val, test = split_labeled_pixels(labels, train_ratio=0.2, val_ratio=0.1, seed=0)

    train, val, test = _as_set(train), _as_set(val), _as_set(test)
    assert not (train & val) and not (train & test) and not (val & test)
    assert train | val | test == _as_set(np.argwhere(labels > 0))


def test_split_keeps_every_class_in_every_set(synthetic_scene):
    _, labels = synthetic_scene
    for coords in split_labeled_pixels(labels, train_ratio=0.2, val_ratio=0.1, seed=0):
        classes = set(labels[coords[:, 0], coords[:, 1]].tolist())
        assert classes == {1, 2, 3}


def test_split_is_reproducible(synthetic_scene):
    _, labels = synthetic_scene
    first = split_labeled_pixels(labels, seed=0)
    second = split_labeled_pixels(labels, seed=0)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


# --- Patch dataset ---

def test_patches_are_centered_on_the_pixel(synthetic_scene):
    cube, labels = synthetic_scene
    # Includes pixels on the image border, where the padding is used
    coords = np.array([[3, 0], [29, 29], [15, 15]])
    dataset = PatchDataset(cube, labels, coords, patch_size=5)

    assert len(dataset) == 3
    for i, (row, col) in enumerate(coords):
        patch, target = dataset[i]
        assert patch.shape == (BANDS, 5, 5)
        assert patch.dtype == torch.float32
        torch.testing.assert_close(patch[:, 2, 2], torch.from_numpy(cube[row, col]))
        assert target.item() == labels[row, col] - 1  # classes start from 0


def test_patch_size_must_be_odd(synthetic_scene):
    cube, labels = synthetic_scene
    with pytest.raises(ValueError):
        PatchDataset(cube, labels, np.array([[5, 5]]), patch_size=4)


# --- Model ---

@pytest.mark.parametrize("patch_size", [3, 7, 11])
def test_cnn2d_output_shape(patch_size):
    model = CNN2D(in_channels=BANDS, num_classes=NUM_CLASSES)
    out = model(torch.randn(4, BANDS, patch_size, patch_size))
    assert out.shape == (4, NUM_CLASSES)


# --- Metrics ---

def test_metrics_known_values():
    # Class 0: 2/2 correct, class 1: 1/2 correct
    metrics = compute_metrics(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 0]))
    assert metrics["OA"] == pytest.approx(0.75)
    assert metrics["AA"] == pytest.approx(0.75)
    assert metrics["Kappa"] == pytest.approx(0.5)
    np.testing.assert_allclose(metrics["per_class"], [1.0, 0.5])


def test_metrics_perfect_prediction():
    y = np.array([0, 1, 2, 2, 1])
    metrics = compute_metrics(y, y)
    assert metrics["OA"] == metrics["AA"] == metrics["Kappa"] == pytest.approx(1.0)


# --- Full pipeline: split -> patches -> CNN -> training -> prediction -> metrics ---

def test_training_pipeline_learns_synthetic_data(synthetic_scene):
    torch.manual_seed(0)
    cube, labels = synthetic_scene
    train, val, test = split_labeled_pixels(labels, train_ratio=0.2, val_ratio=0.1, seed=0)

    train_loader = DataLoader(PatchDataset(cube, labels, train, 5), batch_size=32, shuffle=True)
    val_loader = DataLoader(PatchDataset(cube, labels, val, 5), batch_size=32)
    test_loader = DataLoader(PatchDataset(cube, labels, test, 5), batch_size=32)

    model = CNN2D(in_channels=BANDS, num_classes=NUM_CLASSES)
    model, history = train_model(model, train_loader, val_loader, epochs=5, lr=1e-3)

    assert len(history) == 5
    assert history[-1]["train_loss"] < history[0]["train_loss"]

    y_true, y_pred = predict(model, test_loader)
    assert len(y_true) == len(y_pred) == len(test)
    assert compute_metrics(y_true, y_pred)["OA"] > 0.9
