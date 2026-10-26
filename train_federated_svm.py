"""Train a federated linear SVM matching the final MSNet evaluator."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.svm import SVC

from data.preprocessing.tabular_preprocessor import TabularPreprocessor


LOGGER = logging.getLogger("MSNet.FederatedSVM")
EXCLUDE_COLUMNS = ["EDSS", "Disease duration"]
CLASS_NAMES = ["Healthy Control", "RRMS", "Progressive MS"]
DEFAULT_CANDIDATES = (3.0, 10.83, 30.0)


def _load_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    if frame.shape[1] <= 1:
        frame = pd.read_csv(path, sep=",")
    frame.columns = frame.columns.str.strip()
    return frame


def _partition_by_site(site_codes: np.ndarray) -> List[np.ndarray]:
    site_codes = np.asarray(site_codes)
    if site_codes.ndim != 1 or len(site_codes) == 0:
        raise ValueError("Institution codes must be a non-empty one-dimensional array")
    if pd.isna(site_codes).any():
        raise ValueError("Institution codes cannot be missing for federated clients")
    return [
        np.flatnonzero(site_codes == site)
        for site in np.sort(np.unique(site_codes))
    ]


def federated_linear_svc(
    features: np.ndarray,
    labels: np.ndarray,
    site_codes: np.ndarray,
    c_value: float,
    tolerance: float = 1e-6,
    max_rounds: int = 50,
    probability: bool = False,
) -> Tuple[SVC, Dict[str, Any]]:
    """Fit a linear SVC by exchanging local support vectors and margin violators.

    Each client retains its full local matrix. The coordinator starts from client
    support vectors, then requests only examples inside the current pairwise
    margin. Training stops when no omitted example can affect the global SVM
    optimum, or fails explicitly if the round limit is reached.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("Features and labels must be aligned 2D/1D arrays")
    if not np.isfinite(features).all():
        raise ValueError("Features must be finite")
    if c_value <= 0 or tolerance < 0 or max_rounds < 1:
        raise ValueError("C and max_rounds must be positive; tolerance non-negative")

    clients = _partition_by_site(site_codes)
    classes = np.unique(labels)
    if len(classes) < 2:
        raise ValueError("At least two classes are required to train an SVM")
    class_pairs = [
        (classes[first], classes[second])
        for first in range(len(classes))
        for second in range(first + 1, len(classes))
    ]

    candidates = set()
    for client_indices in clients:
        for first_class, second_class in class_pairs:
            pair_indices = client_indices[
                np.isin(labels[client_indices], [first_class, second_class])
            ]
            if len(np.unique(labels[pair_indices])) < 2:
                continue
            local_model = SVC(C=c_value, kernel="linear")
            local_model.fit(features[pair_indices], labels[pair_indices])
            candidates.update(pair_indices[local_model.support_].tolist())

    if not candidates:
        raise RuntimeError("No client support vectors were produced")

    rounds_completed = 0
    converged = False
    for round_number in range(1, max_rounds + 1):
        newly_requested = set()
        for first_class, second_class in class_pairs:
            pair_candidates = np.asarray(
                sorted(
                    index
                    for index in candidates
                    if labels[index] == first_class or labels[index] == second_class
                ),
                dtype=np.int64,
            )
            pair_model = SVC(C=c_value, kernel="linear")
            pair_model.fit(features[pair_candidates], labels[pair_candidates])
            positive_class = pair_model.classes_[1]

            for client_indices in clients:
                pair_indices = client_indices[
                    np.isin(labels[client_indices], [first_class, second_class])
                ]
                omitted_indices = np.asarray(
                    [index for index in pair_indices if index not in candidates],
                    dtype=np.int64,
                )
                if not len(omitted_indices):
                    continue
                margins = pair_model.decision_function(features[omitted_indices])
                margins *= np.where(
                    labels[omitted_indices] == positive_class, 1.0, -1.0
                )
                newly_requested.update(
                    omitted_indices[margins <= 1.0 + tolerance].tolist()
                )

        rounds_completed = round_number
        if not newly_requested:
            converged = True
            break
        candidates.update(newly_requested)

    if not converged:
        raise RuntimeError(
            f"Federated SVM did not converge within {max_rounds} rounds "
            f"({len(candidates)} support/margin examples collected)"
        )

    candidate_indices = np.asarray(sorted(candidates), dtype=np.int64)
    model = SVC(
        C=c_value,
        kernel="linear",
        probability=probability,
        random_state=42,
    )
    model.fit(features[candidate_indices], labels[candidate_indices])
    return model, {
        "clients": len(clients),
        "client_samples": [int(len(indices)) for indices in clients],
        "rounds": rounds_completed,
        "examples_shared": int(len(candidate_indices)),
        "total_training_examples": int(len(features)),
        "converged": converged,
        "protocol": "iterative pairwise support-vector and margin exchange",
    }


def _metrics(model: SVC, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(
                labels,
                probabilities,
                labels=model.classes_,
                multi_class="ovr",
                average="macro",
            )
        ),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=model.classes_
        ).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=model.classes_,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }


def _fit_and_score(
    features: np.ndarray,
    labels: np.ndarray,
    site_codes: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    c_value: float,
    max_rounds: int,
) -> Tuple[SVC, Dict[str, Any], Dict[str, Any]]:
    model, training_details = federated_linear_svc(
        features,
        labels,
        site_codes,
        c_value=c_value,
        max_rounds=max_rounds,
    )
    predictions = model.predict(validation_features)
    validation_metrics = {
        "accuracy": float(accuracy_score(validation_labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(validation_labels, predictions)
        ),
        "macro_f1": float(
            f1_score(validation_labels, predictions, average="macro", zero_division=0)
        ),
    }
    return model, training_details, validation_metrics


def train_federated_svm(
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    baseline_report_path: Path,
    output_dir: Path,
    c_candidates: Sequence[float] = DEFAULT_CANDIDATES,
    max_rounds: int = 50,
) -> Tuple[Path, Path]:
    train_frame = _load_data(train_path)
    validation_frame = _load_data(validation_path)
    test_frame = _load_data(test_path)

    preprocessor = TabularPreprocessor(
        target_name="phenotype",
        exclude_cols=EXCLUDE_COLUMNS,
        use_feature_engineering=True,
    )
    train_features, train_labels, _ = preprocessor.fit_transform(train_frame)
    validation_features, validation_labels = preprocessor.transform(validation_frame)
    if validation_labels is None:
        raise ValueError("Validation data must include phenotype labels")

    train_sites = pd.to_numeric(
        train_frame["INSTITUTION-CODE"], errors="raise"
    ).to_numpy()
    candidate_runs = []
    for c_value in c_candidates:
        _, details, metrics = _fit_and_score(
            train_features,
            train_labels,
            train_sites,
            validation_features,
            validation_labels,
            c_value=float(c_value),
            max_rounds=max_rounds,
        )
        candidate_runs.append(
            {"C": float(c_value), "training": details, "validation": metrics}
        )
        LOGGER.info(
            "C %.4g | validation accuracy %.4f | macro-F1 %.4f",
            c_value,
            metrics["accuracy"],
            metrics["macro_f1"],
        )

    selected = max(
        candidate_runs,
        key=lambda run: (
            run["validation"]["accuracy"],
            run["validation"]["macro_f1"],
        ),
    )
    selected_c = selected["C"]
    all_features, all_labels = np.vstack(
        [train_features, validation_features]
    ), np.concatenate([train_labels, validation_labels])
    all_sites = np.concatenate(
        [
            train_sites,
            pd.to_numeric(
                validation_frame["INSTITUTION-CODE"], errors="raise"
            ).to_numpy(),
        ]
    )

    final_model, final_training = federated_linear_svc(
        all_features,
        all_labels,
        all_sites,
        c_value=selected_c,
        max_rounds=max_rounds,
        probability=True,
    )
    test_features, test_labels = preprocessor.transform(test_frame)
    if test_labels is None:
        raise ValueError("Test data must include phenotype labels")
    test_metrics = _metrics(final_model, test_features, test_labels)

    baseline = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    baseline_metrics = baseline["classification"]
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "federated_linear_svc.joblib"
    joblib.dump(
        {
            "model": final_model,
            "preprocessor": preprocessor,
            "config": {
                "kernel": "linear",
                "C": selected_c,
                "features": preprocessor.feature_names,
            },
        },
        checkpoint_path,
    )

    report = {
        "model": "Federated linear SVC",
        "reference": str(baseline_report_path),
        "reference_model": "SVC(C=10.83, kernel='linear')",
        "reference_test_accuracy": float(baseline_metrics["accuracy"]),
        "selected_C": selected_c,
        "validation_selection_metric": "accuracy, then macro-F1",
        "validation_candidates": candidate_runs,
        "training_data": str(train_path),
        "validation_data": str(validation_path),
        "test_data": str(test_path),
        "training_recipe": (
            "Same preprocessing, linear SVC objective, and train+validation "
            "cohort as evaluate_final_model.py; site clients exchange support "
            "vectors and margin examples until the global optimum converges."
        ),
        "training": final_training,
        "test_metrics": test_metrics,
        "accuracy_delta_vs_reference": float(
            test_metrics["accuracy"] - baseline_metrics["accuracy"]
        ),
        "privacy_note": (
            "This support-vector exchange simulation does not provide formal "
            "privacy: exchanged training examples can reveal patient information."
        ),
    }
    report_path = output_dir / "federated_svm_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return checkpoint_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a federated SVC using the final MSNet evaluation recipe"
    )
    parser.add_argument("--train-data", type=Path, default=Path("data/train_split.csv"))
    parser.add_argument("--validation-data", type=Path, default=Path("data/val_data.csv"))
    parser.add_argument("--test-data", type=Path, default=Path("data/fp__edss.csv"))
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=Path("outputs/final_evaluation_report.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/federated_svm")
    )
    parser.add_argument(
        "--c-candidates",
        type=float,
        nargs="+",
        default=list(DEFAULT_CANDIDATES),
        help="Regularization candidates selected using validation only",
    )
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for path in (
        args.train_data,
        args.validation_data,
        args.test_data,
        args.baseline_report,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
    checkpoint, report = train_federated_svm(
        train_path=args.train_data,
        validation_path=args.validation_data,
        test_path=args.test_data,
        baseline_report_path=args.baseline_report,
        output_dir=args.output_dir,
        c_candidates=args.c_candidates,
        max_rounds=args.max_rounds,
    )
    print(f"Federated checkpoint: {checkpoint}")
    print(f"Federated report: {report}")


if __name__ == "__main__":
    main()
