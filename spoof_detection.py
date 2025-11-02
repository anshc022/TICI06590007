"""Fingerprint spoof detection using HOG + LBP features and an SVM classifier.

This script expects the Biometrika dataset laid out as:
    <root>/Training Biometrika Live/live
    <root>/Training Biometrika Spoof/Training Biometrika Spoof/spoof
    <root>/Testing Biometrika Live/live
    <root>/Testing Biometrika Spoof/Testing Biometrika Spoof/spoof

It extracts HOG and LBP features, trains an SVM on the training split, and
reports the requested metrics on the held-out test split.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern
from skimage.io import imread
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class DataSplit:
    """Holds flattened feature vectors and labels for a dataset partition."""

    features: np.ndarray
    labels: np.ndarray


def iter_image_files(directory: Path) -> Iterable[Path]:
    """Yield image file paths in a deterministic order."""

    for path in sorted(directory.glob("*.png")):
        if path.is_file():
            yield path


def extract_feature_vector(
    image_path: Path,
    *,
    lbp_radius: int = 3,
    lbp_method: str = "uniform",
    hog_orientations: int = 9,
    hog_pixels_per_cell: Tuple[int, int] = (8, 8),
    hog_cells_per_block: Tuple[int, int] = (2, 2),
) -> np.ndarray:
    """Return concatenated HOG and LBP features for a single image."""

    image = imread(image_path)
    gray_image = rgb2gray(image)

    lbp_points = 8 * lbp_radius
    lbp = local_binary_pattern(gray_image, lbp_points, lbp_radius, lbp_method)
    lbp_features = lbp.flatten()

    hog_features, _ = hog(
        gray_image,
        orientations=hog_orientations,
        pixels_per_cell=hog_pixels_per_cell,
        cells_per_block=hog_cells_per_block,
        visualize=True,
        channel_axis=None,
        block_norm="L2-Hys",
    )

    return np.concatenate([hog_features, lbp_features])


def build_split(live_dir: Path, spoof_dir: Path) -> DataSplit:
    """Assemble features and labels for a dataset partition."""

    live_features: List[np.ndarray] = []
    for image_path in iter_image_files(live_dir):
        live_features.append(extract_feature_vector(image_path))

    spoof_features: List[np.ndarray] = []
    for image_path in iter_image_files(spoof_dir):
        spoof_features.append(extract_feature_vector(image_path))

    expected_dim = live_features[0].shape[0] if live_features else None
    for feature_vector in live_features + spoof_features:
        if expected_dim is None:
            expected_dim = feature_vector.shape[0]
        elif feature_vector.shape[0] != expected_dim:
            message = (
                "Feature length mismatch encountered; ensure all images share the same "
                "dimensions before feature extraction."
            )
            raise ValueError(message)

    features = np.vstack([live_features, spoof_features]).astype(np.float32)
    labels = np.concatenate(
        [np.zeros(len(live_features), dtype=np.int8), np.ones(len(spoof_features), dtype=np.int8)]
    )

    return DataSplit(features=features, labels=labels)


def make_model(*, c: float, gamma: str, kernel: str) -> Pipeline:
    """Create the SVM pipeline."""

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    C=c,
                    kernel=kernel,
                    gamma=gamma,
                    class_weight="balanced",
                    probability=False,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_predictions(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """Compute all required metrics given true and predicted labels."""

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "accuracy": float(accuracy),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "matthews_corrcoef": float(mcc),
    }


def train_and_evaluate(data_root: Path, *, c: float, gamma: str, kernel: str) -> dict:
    """Train the SVM on the training split and evaluate on the test split."""

    train_live = data_root / "Training Biometrika Live" / "live"
    train_spoof = data_root / "Training Biometrika Spoof" / "Training Biometrika Spoof" / "spoof"

    test_live = data_root / "Testing Biometrika Live" / "live"
    test_spoof = data_root / "Testing Biometrika Spoof" / "Testing Biometrika Spoof" / "spoof"

    for directory in (train_live, train_spoof, test_live, test_spoof):
        if not directory.exists():
            message = f"Expected directory missing: {directory}"
            raise FileNotFoundError(message)

    train_split = build_split(train_live, train_spoof)
    test_split = build_split(test_live, test_spoof)

    model = make_model(c=c, gamma=gamma, kernel=kernel)
    model.fit(train_split.features, train_split.labels)

    predictions = model.predict(test_split.features)
    metrics = evaluate_predictions(test_split.labels, predictions)
    metrics["num_train_samples"] = int(train_split.labels.size)
    metrics["num_test_samples"] = int(test_split.labels.size)

    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("Spoof_data"),
        help="Path to the Spoof_data directory containing the train/test folders.",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        choices=("linear", "rbf"),
        help="SVM kernel to use.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Inverse regularisation strength (C) for the SVM.",
    )
    parser.add_argument(
        "--gamma",
        type=str,
        default="scale",
        help="Gamma parameter for the SVM kernel (as accepted by sklearn).",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Optional path to write the evaluation metrics as JSON.",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    metrics = train_and_evaluate(args.data_root, c=args.c, gamma=args.gamma, kernel=args.kernel)

    print("Evaluation metrics:\n")
    print(json.dumps(metrics, indent=2))

    if args.metrics_json:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
